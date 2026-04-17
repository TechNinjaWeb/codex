use std::sync::Arc;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::compact::InitialContextInjection;
use crate::compact::insert_initial_context_before_last_real_user_or_summary;
use crate::context_manager::compute_protected_tail_split;
use crate::context_manager::validate_replacement_history;
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

const MIN_LCM_COMPACTION_SOURCE_TOKENS: u32 = 4_096;
const MIN_LCM_COMPACTION_SAVINGS_TOKENS: u32 = 2_048;

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
    if let Some(reason) = lcm_compaction_skip_reason(source_token_count, summary_token_count) {
        tracing::info!(
            source_token_count,
            summary_token_count,
            reason,
            "skipping low-value Open Brain LCM compaction"
        );
        sess.notify_background_event(
            turn_context.as_ref(),
            format!("LCM compaction skipped: {reason}."),
        )
        .await;
        return Ok(());
    }

    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;

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

    crate::lcm_durable_memory::promote_from_compaction(
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
    if let Err(issues) = validate_replacement_history(&new_history) {
        return Err(CodexErr::Fatal(format!(
            "Open Brain LCM replacement history failed integrity validation: {issues:?}"
        )));
    }

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

fn lcm_compaction_skip_reason(source_token_count: u32, summary_token_count: u32) -> Option<String> {
    if source_token_count == 0 {
        return Some("nothing beyond the fresh tail is compactable".to_string());
    }

    if source_token_count < MIN_LCM_COMPACTION_SOURCE_TOKENS {
        return Some(format!(
            "source head is only {source_token_count} tokens; minimum useful compaction size is {MIN_LCM_COMPACTION_SOURCE_TOKENS}"
        ));
    }

    let estimated_savings = source_token_count.saturating_sub(summary_token_count);
    if estimated_savings < MIN_LCM_COMPACTION_SAVINGS_TOKENS {
        return Some(format!(
            "projected savings are only {estimated_savings} tokens; minimum useful savings are {MIN_LCM_COMPACTION_SAVINGS_TOKENS}"
        ));
    }

    None
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
    let split_at = compute_protected_tail_split(items, fresh_tail_count);
    (items[..split_at].to_vec(), items[split_at..].to_vec())
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

#[cfg(test)]
mod tests {
    use super::context_history_message;
    use super::lcm_compaction_skip_reason;
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
    fn split_history_keeps_separated_tool_pairs_in_tail() {
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
            ResponseItem::Reasoning {
                id: String::new(),
                summary: Vec::new(),
                encrypted_content: None,
                content: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ];

        let (head, tail) = split_history_for_lcm(&items, 1);
        assert_eq!(head.len(), 1);
        assert_eq!(tail.len(), 3);
        assert!(matches!(tail[0], ResponseItem::FunctionCall { .. }));
        assert!(matches!(tail[1], ResponseItem::Reasoning { .. }));
        assert!(matches!(tail[2], ResponseItem::FunctionCallOutput { .. }));
    }

    #[test]
    fn compaction_skip_reason_rejects_tiny_head() {
        assert_eq!(
            lcm_compaction_skip_reason(2_628, 1_207),
            Some(
                "source head is only 2628 tokens; minimum useful compaction size is 4096"
                    .to_string()
            )
        );
    }

    #[test]
    fn compaction_skip_reason_rejects_low_savings() {
        assert_eq!(
            lcm_compaction_skip_reason(4_800, 3_000),
            Some(
                "projected savings are only 1800 tokens; minimum useful savings are 2048"
                    .to_string()
            )
        );
    }

    #[test]
    fn compaction_skip_reason_allows_material_compaction() {
        assert_eq!(lcm_compaction_skip_reason(8_300, 1_207), None);
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
}
