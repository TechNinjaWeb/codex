use std::sync::Arc;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::compact::InitialContextInjection;
use crate::compact::insert_initial_context_before_last_real_user_or_summary;
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
            codex_open_brain::ContextPacketItemKind::GraphFrontier => Some(history_message(
                "LCM frontier summary",
                item.content.clone(),
                item.selection_reason.clone(),
            )),
            codex_open_brain::ContextPacketItemKind::DurableMemory => Some(history_message(
                "LCM durable memory",
                item.content.clone(),
                item.selection_reason.clone(),
            )),
            codex_open_brain::ContextPacketItemKind::ExplicitExpansion => Some(history_message(
                "LCM expansion",
                item.content.clone(),
                item.selection_reason.clone(),
            )),
            codex_open_brain::ContextPacketItemKind::WorkingMemory => Some(history_message(
                "LCM working memory",
                item.content.clone(),
                item.selection_reason.clone(),
            )),
            codex_open_brain::ContextPacketItemKind::SystemInstruction => Some(history_message(
                "LCM system instruction",
                item.content.clone(),
                item.selection_reason.clone(),
            )),
        })
        .collect()
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
    use super::split_history_for_lcm;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;

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
}
