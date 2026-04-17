use std::sync::Arc;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::compact::InitialContextInjection;
use crate::compact::insert_initial_context_before_last_real_user_or_summary;
use codex_protocol::ThreadId;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::CompactedItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_output_truncation::truncate_text;

pub(crate) async fn run_inline_lcm_auto_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    initial_context_injection: InitialContextInjection,
) -> CodexResult<()> {
    run_lcm_compact_task_inner(sess, turn_context, Vec::new(), initial_context_injection).await
}

pub(crate) async fn run_lcm_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
) -> CodexResult<()> {
    let start_event = EventMsg::TurnStarted(TurnStartedEvent {
        turn_id: turn_context.sub_id.clone(),
        started_at: turn_context.turn_timing_state.started_at_unix_secs().await,
        model_context_window: turn_context.model_context_window(),
        collaboration_mode_kind: turn_context.collaboration_mode.mode,
    });
    sess.send_event(&turn_context, start_event).await;
    run_lcm_compact_task_inner(
        sess,
        turn_context,
        input,
        InitialContextInjection::DoNotInject,
    )
    .await
}

async fn run_lcm_compact_task_inner(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
    initial_context_injection: InitialContextInjection,
) -> CodexResult<()> {
    let Some(runtime) = sess.open_brain_runtime().await else {
        return Err(CodexErr::UnsupportedOperation(
            "open_brain_lcm selected without Open Brain runtime".to_string(),
        ));
    };

    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;

    let history_snapshot = sess.clone_history().await;
    let history_items = history_snapshot.raw_items();
    let (head, tail) = split_history_for_lcm(history_items, runtime.lcm().fresh_tail_count);
    let source_event_refs = runtime.source_event_refs_for_items(&head);
    let head_text = render_history_block(&head);
    let source_token_count = approx_token_count(&head_text) as u32;
    let summary_text = if head_text.trim().is_empty() {
        String::new()
    } else {
        truncate_text(
            &head_text,
            TruncationPolicy::Tokens(runtime.lcm().leaf_target_tokens),
        )
    };
    let summary_token_count = approx_token_count(&summary_text) as u32;
    let leaf_summary = if !summary_text.trim().is_empty() {
        Some(
            runtime
                .compact_leaf(
                    &summary_text,
                    &source_event_refs,
                    source_token_count,
                    summary_token_count,
                )
                .await
                .map_err(|err| {
                    CodexErr::Fatal(format!("Open Brain leaf compaction failed: {err}"))
                })?,
        )
    } else {
        None
    };

    maybe_promote_lcm_durable_memory(
        sess.as_ref(),
        turn_context.as_ref(),
        &runtime,
        &head,
        &source_event_refs,
        leaf_summary.as_ref(),
        &summary_text,
        source_token_count,
        summary_token_count,
    )
    .await;

    let packet = runtime
        .assemble_context(
            lcm_query(&input, history_items).as_deref(),
            runtime.context_token_budget(turn_context.model_context_window()),
        )
        .await
        .map_err(|err| CodexErr::Fatal(format!("Open Brain context assembly failed: {err}")))?;

    sess.persist_open_brain_packet_metadata(packet.packet_id.clone())
        .await;

    let mut new_history = packet_items_to_history(&packet);
    if new_history.is_empty() && !summary_text.trim().is_empty() {
        new_history.push(history_message(
            "LCM leaf summary",
            summary_text.clone(),
            "active compressed history",
        ));
    }
    if matches!(
        initial_context_injection,
        InitialContextInjection::BeforeLastUserMessage
    ) {
        let initial_context = sess.build_initial_context(turn_context.as_ref()).await;
        new_history =
            insert_initial_context_before_last_real_user_or_summary(new_history, initial_context);
    }
    new_history.extend(tail.clone());

    let reference_context_item = match initial_context_injection {
        InitialContextInjection::DoNotInject => None,
        InitialContextInjection::BeforeLastUserMessage => Some(turn_context.to_turn_context_item()),
    };
    let compacted_item = CompactedItem {
        message: render_packet_message(&packet, &summary_text),
        replacement_history: Some(new_history.clone()),
        summary_node_id: leaf_summary
            .as_ref()
            .map(|summary| summary.summary_node_id.clone()),
        span_id: leaf_summary
            .as_ref()
            .and_then(|summary| summary.span_id.clone()),
        depth: Some(0),
        src_tok: (source_token_count > 0).then_some(source_token_count),
        desc_tok: (summary_token_count > 0).then_some(summary_token_count),
        fresh_tail_count: Some(tail.len() as u32),
    };
    sess.replace_compacted_history(new_history, reference_context_item, compacted_item)
        .await;
    sess.recompute_token_usage(&turn_context).await;
    sess.emit_turn_item_completed(&turn_context, compaction_item)
        .await;
    Ok(())
}

fn lcm_query(input: &[UserInput], history_items: &[ResponseItem]) -> Option<String> {
    let joined = input
        .iter()
        .filter_map(|item| match item {
            UserInput::Text { text, .. } => {
                let trimmed = text.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !joined.trim().is_empty() {
        return Some(joined);
    }
    history_items.iter().rev().find_map(|item| match item {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            let text = content_items_to_text(content);
            text.filter(|text| !text.trim().is_empty())
        }
        _ => None,
    })
}

fn split_history_for_lcm(
    items: &[ResponseItem],
    fresh_tail_count: usize,
) -> (Vec<ResponseItem>, Vec<ResponseItem>) {
    if items.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut split_at = items
        .len()
        .saturating_sub(fresh_tail_count.min(items.len()));
    while split_at > 0 && should_expand_tail_left(items.get(split_at - 1), items.get(split_at)) {
        split_at -= 1;
    }
    (items[..split_at].to_vec(), items[split_at..].to_vec())
}

fn should_expand_tail_left(left: Option<&ResponseItem>, right: Option<&ResponseItem>) -> bool {
    matches!(
        (left, right),
        (
            Some(
                ResponseItem::FunctionCall { .. }
                    | ResponseItem::CustomToolCall { .. }
                    | ResponseItem::ToolSearchCall { .. }
            ),
            Some(
                ResponseItem::FunctionCallOutput { .. }
                    | ResponseItem::CustomToolCallOutput { .. }
                    | ResponseItem::ToolSearchOutput { .. }
            )
        )
    )
}

fn render_history_block(items: &[ResponseItem]) -> String {
    items
        .iter()
        .filter_map(render_history_item)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_history_item(item: &ResponseItem) -> Option<String> {
    match item {
        ResponseItem::Message { role, content, .. } => {
            content_items_to_text(content).map(|text| format!("{role}: {text}"))
        }
        ResponseItem::FunctionCall {
            name, arguments, ..
        } => Some(format!("assistant tool call {name}: {arguments}")),
        ResponseItem::FunctionCallOutput { output, .. } => Some(format!("tool output: {output}")),
        ResponseItem::CustomToolCall { name, input, .. } => {
            Some(format!("assistant custom tool {name}: {input}"))
        }
        ResponseItem::CustomToolCallOutput { output, .. } => Some(format!("tool output: {output}")),
        ResponseItem::ToolSearchCall {
            execution,
            arguments,
            ..
        } => Some(format!("assistant tool search {execution}: {arguments}")),
        ResponseItem::ToolSearchOutput {
            status, execution, ..
        } => Some(format!("tool search {status}: {execution}")),
        ResponseItem::Reasoning { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::GhostSnapshot { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::Other => None,
    }
}

fn content_items_to_text(items: &[ContentItem]) -> Option<String> {
    let joined = items
        .iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                let trimmed = text.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            ContentItem::InputImage { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!joined.trim().is_empty()).then_some(joined)
}

fn packet_items_to_history(packet: &codex_open_brain::ContextPacket) -> Vec<ResponseItem> {
    packet
        .items
        .iter()
        .filter_map(|item| match item.kind {
            codex_open_brain::ContextPacketItemKind::RecentTail => None,
            codex_open_brain::ContextPacketItemKind::GraphFrontier => {
                Some(context_history_message(
                    "LCM frontier summary",
                    item.content.clone(),
                    item.selection_reason.clone(),
                ))
            }
            codex_open_brain::ContextPacketItemKind::DurableMemory => {
                Some(context_history_message(
                    "LCM durable memory",
                    item.content.clone(),
                    item.selection_reason.clone(),
                ))
            }
            codex_open_brain::ContextPacketItemKind::ExplicitExpansion => {
                Some(context_history_message(
                    "LCM expansion",
                    item.content.clone(),
                    item.selection_reason.clone(),
                ))
            }
            codex_open_brain::ContextPacketItemKind::WorkingMemory => {
                Some(context_history_message(
                    "LCM working memory",
                    item.content.clone(),
                    item.selection_reason.clone(),
                ))
            }
            codex_open_brain::ContextPacketItemKind::SystemInstruction => {
                Some(context_history_message(
                    "LCM system instruction",
                    item.content.clone(),
                    item.selection_reason.clone(),
                ))
            }
        })
        .collect()
}

fn context_history_message(
    title: impl AsRef<str>,
    content: String,
    selection_reason: impl AsRef<str>,
) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: format!(
                "{}\n{}\n\nReason: {}",
                title.as_ref(),
                content,
                selection_reason.as_ref()
            ),
        }],
        end_turn: None,
        phase: None,
    }
}

fn history_message(
    title: impl AsRef<str>,
    content: String,
    selection_reason: impl AsRef<str>,
) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: format!(
                "{}\n{}\n\nReason: {}",
                title.as_ref(),
                content,
                selection_reason.as_ref()
            ),
        }],
        end_turn: None,
        phase: None,
    }
}

fn render_packet_message(packet: &codex_open_brain::ContextPacket, summary_text: &str) -> String {
    if !summary_text.trim().is_empty() {
        format!("LCM packet {} with leaf summary", packet.packet_id)
    } else {
        format!("LCM packet {}", packet.packet_id)
    }
}

async fn maybe_promote_lcm_durable_memory(
    sess: &Session,
    turn_context: &TurnContext,
    runtime: &codex_open_brain::OpenBrainRuntime,
    head: &[ResponseItem],
    source_event_refs: &[String],
    leaf_summary: Option<&codex_open_brain::OpenBrainLeafSummary>,
    summary_text: &str,
    source_token_count: u32,
    summary_token_count: u32,
) {
    let Some(leaf_summary) = leaf_summary else {
        tracing::debug!("skipping LCM durable-memory promotion: no leaf summary");
        return;
    };
    let promotion_mode = lcm_memory_promotion_mode(sess, sess.conversation_id).await;
    if !promotion_mode.allows_promotion() {
        tracing::info!(
            summary_node_id = %leaf_summary.summary_node_id,
            memory_mode = %promotion_mode.as_str(),
            "skipping LCM durable-memory promotion: thread memory mode disallows promotion"
        );
        sess.notify_background_event(
            turn_context,
            format!(
                "LCM durable memory skipped because thread memory mode is {}.",
                promotion_mode.as_str()
            ),
        )
        .await;
        return;
    }

    let candidate = codex_open_brain::OpenBrainDurableMemoryCandidate {
        summary_node_id: leaf_summary.summary_node_id.clone(),
        title: latest_user_message_title(head),
        content: summary_text.to_string(),
        source_event_ids: source_event_refs.to_vec(),
        source_token_count: Some(source_token_count),
        summary_token_count: Some(summary_token_count),
    };
    promote_lcm_durable_memory_candidate(
        sess,
        Some(turn_context),
        runtime,
        candidate,
        "active compaction",
    )
    .await;
}

pub(crate) fn start_lcm_durable_memory_backfill_task(sess: &Arc<Session>) {
    let sess = Arc::clone(sess);
    tokio::spawn(async move {
        if let Err(err) = backfill_latest_lcm_durable_memory(sess).await {
            tracing::warn!("failed to backfill LCM durable memory on startup: {err}");
        }
    });
}

async fn backfill_latest_lcm_durable_memory(sess: Arc<Session>) -> CodexResult<()> {
    let Some(runtime) = sess.open_brain_runtime().await else {
        return Ok(());
    };

    let promotion_mode = lcm_memory_promotion_mode(sess.as_ref(), sess.conversation_id).await;
    if !promotion_mode.allows_promotion() {
        tracing::debug!(
            memory_mode = %promotion_mode.as_str(),
            "skipping LCM durable-memory backfill: thread memory mode disallows promotion"
        );
        return Ok(());
    }

    let Some(candidate) = runtime
        .latest_unpromoted_summary_candidate(128)
        .await
        .map_err(|err| {
            CodexErr::Fatal(format!("Open Brain durable-memory backfill failed: {err}"))
        })?
    else {
        tracing::debug!("skipping LCM durable-memory backfill: no unpromoted summary found");
        return Ok(());
    };

    promote_lcm_durable_memory_candidate(
        sess.as_ref(),
        None,
        &runtime,
        candidate,
        "startup backfill",
    )
    .await;
    Ok(())
}

async fn promote_lcm_durable_memory_candidate(
    sess: &Session,
    turn_context: Option<&TurnContext>,
    runtime: &codex_open_brain::OpenBrainRuntime,
    candidate: codex_open_brain::OpenBrainDurableMemoryCandidate,
    promotion_source: &str,
) {
    if candidate.content.trim().is_empty() {
        tracing::debug!(
            summary_node_id = %candidate.summary_node_id,
            promotion_source,
            "skipping LCM durable-memory promotion: empty summary text"
        );
        return;
    }

    let source_token_count = candidate.source_token_count.unwrap_or_default();

    let record = codex_open_brain::OpenBrainDurableMemoryRecord {
        title: candidate.title,
        content: candidate.content,
        memory_type: "lcm_leaf_summary".to_string(),
        source_node_ids: vec![candidate.summary_node_id.clone()],
        source_event_ids: candidate.source_event_ids,
        metadata: std::collections::BTreeMap::from([
            (
                "memory_kind".to_string(),
                serde_json::Value::String("lcm_leaf_summary".to_string()),
            ),
            (
                "summary_node_id".to_string(),
                serde_json::Value::String(candidate.summary_node_id.clone()),
            ),
            (
                "source_token_count".to_string(),
                serde_json::Value::from(source_token_count),
            ),
            (
                "summary_token_count".to_string(),
                serde_json::Value::from(candidate.summary_token_count.unwrap_or_default()),
            ),
            (
                "promotion_source".to_string(),
                serde_json::Value::String(promotion_source.to_string()),
            ),
        ]),
        mirror_to_thoughts: runtime.mirror_durable_to_thoughts(),
    };
    match runtime.promote_durable_memory(&[record]).await {
        Ok(result) => {
            tracing::info!(
                summary_node_id = %candidate.summary_node_id,
                inserted_count = result.inserted_count,
                promotion_source,
                "Open Brain durable-memory promotion succeeded"
            );
            if result.inserted_count > 0 {
                if let Some(turn_context) = turn_context {
                    sess.notify_background_event(
                        turn_context,
                        format!(
                            "LCM durable memory promoted from summary {}.",
                            candidate.summary_node_id
                        ),
                    )
                    .await;
                }
            }
        }
        Err(err) => {
            tracing::warn!(
                summary_node_id = %candidate.summary_node_id,
                promotion_source,
                "Open Brain durable-memory promotion failed: {err}"
            );
            if let Some(turn_context) = turn_context {
                sess.notify_background_event(
                    turn_context,
                    format!("LCM durable-memory promotion failed: {err}"),
                )
                .await;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LcmMemoryPromotionMode {
    Enabled,
    Disabled,
    Legacy(String),
}

impl LcmMemoryPromotionMode {
    fn allows_promotion(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::Legacy(mode) => mode.as_str(),
        }
    }
}

fn lcm_resolve_memory_promotion_mode(mode: Option<&str>) -> LcmMemoryPromotionMode {
    match mode {
        Some("disabled") => LcmMemoryPromotionMode::Disabled,
        Some("enabled") | None => LcmMemoryPromotionMode::Enabled,
        Some(legacy) => LcmMemoryPromotionMode::Legacy(legacy.to_string()),
    }
}

async fn lcm_memory_promotion_mode(sess: &Session, thread_id: ThreadId) -> LcmMemoryPromotionMode {
    let Some(state_db) = sess.services.state_db.as_deref() else {
        return LcmMemoryPromotionMode::Enabled;
    };
    match state_db.get_thread_memory_mode(thread_id).await {
        Ok(mode) => lcm_resolve_memory_promotion_mode(mode.as_deref()),
        Err(err) => {
            tracing::warn!("failed to read thread memory mode for LCM promotion: {err}");
            LcmMemoryPromotionMode::Enabled
        }
    }
}

fn latest_user_message_title(items: &[ResponseItem]) -> String {
    latest_user_message_text(items)
        .map(|text| format!("LCM memory: {}", preview_text(&text, 72)))
        .unwrap_or_else(|| "LCM durable memory".to_string())
}

fn latest_user_message_text(items: &[ResponseItem]) -> Option<String> {
    items.iter().rev().find_map(|item| match item {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content)
        }
        _ => None,
    })
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        preview.push('…');
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::LcmMemoryPromotionMode;
    use super::context_history_message;
    use super::latest_user_message_title;
    use super::lcm_resolve_memory_promotion_mode;
    use super::packet_items_to_history;
    use super::split_history_for_lcm;
    use codex_open_brain::ContextPacket;
    use codex_open_brain::ContextPacketItem;
    use codex_open_brain::ContextPacketItemKind;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;

    #[test]
    fn split_history_keeps_tool_pairs_in_tail() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![codex_protocol::models::ContentItem::InputText {
                    text: "before".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "tool".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call-1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ];

        let (head, tail) = split_history_for_lcm(&items, 1);
        assert_eq!(head.len(), 1);
        assert_eq!(tail.len(), 2);
        assert!(matches!(tail[0], ResponseItem::FunctionCall { .. }));
        assert!(matches!(tail[1], ResponseItem::FunctionCallOutput { .. }));
    }

    #[test]
    fn latest_user_message_title_uses_recent_user_prompt() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![codex_protocol::models::ContentItem::InputText {
                    text: "Initial task".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![codex_protocol::models::ContentItem::OutputText {
                    text: "Working".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![codex_protocol::models::ContentItem::InputText {
                    text: "Capture the project-scoped decision and preserve the context"
                        .to_string(),
                }],
                end_turn: None,
                phase: None,
            },
        ];

        assert_eq!(
            latest_user_message_title(&items),
            "LCM memory: Capture the project-scoped decision and preserve the context"
        );
    }

    #[test]
    fn packet_history_replays_as_assistant_markdown() {
        let packet = ContextPacket {
            packet_id: "packet-1".to_string(),
            packet_node_id: None,
            session_id: None,
            thread_id: "thread-1".to_string(),
            project_key: None,
            scope_key: None,
            token_budget: 4000,
            items: vec![
                ContextPacketItem {
                    kind: ContextPacketItemKind::GraphFrontier,
                    content: "**Strong**\n- bullet".to_string(),
                    source_refs: Vec::new(),
                    selection_reason: "frontier".to_string(),
                    depth: Some(0),
                    src_tok: Some(100),
                    desc_tok: Some(20),
                },
                ContextPacketItem {
                    kind: ContextPacketItemKind::RecentTail,
                    content: "leave tail alone".to_string(),
                    source_refs: Vec::new(),
                    selection_reason: "tail".to_string(),
                    depth: None,
                    src_tok: None,
                    desc_tok: None,
                },
            ],
        };

        let history = packet_items_to_history(&packet);
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0],
            context_history_message(
                "LCM frontier summary",
                "**Strong**\n- bullet".to_string(),
                "frontier",
            )
        );

        let ResponseItem::Message { role, content, .. } = &history[0] else {
            panic!("expected message response item");
        };
        assert_eq!(role, "assistant");
        assert_eq!(
            content,
            &vec![ContentItem::OutputText {
                text: "LCM frontier summary\n**Strong**\n- bullet\n\nReason: frontier".to_string(),
            }]
        );
    }

    #[test]
    fn lcm_memory_promotion_blocks_only_disabled_mode() {
        assert_eq!(
            lcm_resolve_memory_promotion_mode(Some("disabled")),
            LcmMemoryPromotionMode::Disabled
        );
        assert!(!lcm_resolve_memory_promotion_mode(Some("disabled")).allows_promotion());
        assert!(lcm_resolve_memory_promotion_mode(Some("enabled")).allows_promotion());
        assert!(lcm_resolve_memory_promotion_mode(None).allows_promotion());
    }

    #[test]
    fn lcm_memory_promotion_allows_legacy_modes() {
        assert_eq!(
            lcm_resolve_memory_promotion_mode(Some("polluted")),
            LcmMemoryPromotionMode::Legacy("polluted".to_string())
        );
        assert!(lcm_resolve_memory_promotion_mode(Some("polluted")).allows_promotion());
    }
}
