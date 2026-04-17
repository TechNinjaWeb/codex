use std::collections::BTreeMap;
use std::sync::Arc;

use crate::Prompt;
use crate::codex::Session;
use crate::codex::TurnContext;
use codex_api::ResponseEvent;
use codex_open_brain::OpenBrainDurableMemoryRecord;
use codex_open_brain::OpenBrainProjectMemory;
use codex_open_brain::OpenBrainRuntime;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::config_types::ServiceTier;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_secrets::redact_secrets;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use tracing::info;
use tracing::warn;

const DURABLE_MEMORY_QUALITY_VERSION: u32 = 1;
const MAX_PROMOTED_RECORDS: usize = 2;
const MAX_UPGRADE_BATCH: usize = 8;
const EXTRACTION_EVIDENCE_TOKEN_BUDGET: usize = 700;
const CONFIDENCE_THRESHOLD: f64 = 0.72;
const EXTRACTION_PROMPT: &str = r#"Extract durable project memory records from the provided source.

Return JSON only.

Rules:
- Output at most 2 records.
- Output nothing if the source is only conversational recap, weak research trivia, or boilerplate.
- Prefer durable project memory that will help a new thread resume work correctly.
- Allowed memory types:
  - decision: a stable implementation or product decision
  - constraint: a required limitation, rule, or invariant
  - known_issue: an active bug, gap, or risk that must stay visible
  - implementation_status: a concise statement of shipped, partial, or deferred state
  - operator_preference: a stable user workflow or behavior preference
- Titles must be short, standalone statements, not questions.
- Content must be 1-3 short sentences in independent language.
- Do not copy system/context wrappers, prompt boilerplate, or raw transcript fragments.
- Do not emit titles like durable memory, LCM memory, summary, note, or recap.
"#;

#[derive(Clone, Debug)]
struct RequestContext {
    model_info: ModelInfo,
    session_telemetry: SessionTelemetry,
    reasoning_effort: Option<ReasoningEffortConfig>,
    reasoning_summary: ReasoningSummaryConfig,
    service_tier: Option<ServiceTier>,
    turn_metadata_header: Option<String>,
}

impl RequestContext {
    fn from_turn_context(turn_context: &TurnContext) -> Self {
        Self {
            model_info: turn_context.model_info.clone(),
            session_telemetry: turn_context.session_telemetry.clone(),
            reasoning_effort: turn_context.reasoning_effort,
            reasoning_summary: turn_context.reasoning_summary,
            service_tier: turn_context.config.service_tier,
            turn_metadata_header: turn_context.turn_metadata_state.current_header_value(),
        }
    }
}

#[derive(Clone, Debug)]
struct DurableMemoryExtractionInput {
    source_summary_node_id: Option<String>,
    latest_user_message: Option<String>,
    summary_text: String,
    evidence_text: Option<String>,
    source_node_ids: Vec<String>,
    source_event_ids: Vec<String>,
    source_token_count: Option<u32>,
    summary_token_count: Option<u32>,
    promotion_source: &'static str,
    supersede_node_ids: Vec<String>,
    mirror_to_thoughts: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DurableMemoryModelOutput {
    records: Vec<DurableMemoryModelRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DurableMemoryModelRecord {
    memory_type: String,
    title: String,
    content: String,
    confidence: f64,
}

pub(crate) async fn promote_from_compaction(
    sess: &Session,
    turn_context: &TurnContext,
    runtime: &OpenBrainRuntime,
    head: &[ResponseItem],
    source_event_ids: &[String],
    leaf_summary: Option<&codex_open_brain::OpenBrainLeafSummary>,
    summary_text: &str,
    source_token_count: u32,
    summary_token_count: u32,
) {
    let Some(leaf_summary) = leaf_summary else {
        tracing::debug!("skipping LCM durable-memory promotion: no leaf summary");
        return;
    };
    let promotion_mode = memory_promotion_mode(sess, sess.conversation_id).await;
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

    let request = RequestContext::from_turn_context(turn_context);
    let input = DurableMemoryExtractionInput {
        source_summary_node_id: Some(leaf_summary.summary_node_id.clone()),
        latest_user_message: latest_user_message_text(head),
        summary_text: summary_text.to_string(),
        evidence_text: evidence_text(head),
        source_node_ids: vec![leaf_summary.summary_node_id.clone()],
        source_event_ids: source_event_ids.to_vec(),
        source_token_count: Some(source_token_count),
        summary_token_count: Some(summary_token_count),
        promotion_source: "active compaction",
        supersede_node_ids: Vec::new(),
        mirror_to_thoughts: runtime.mirror_durable_to_thoughts(),
    };
    promote_records(sess, Some(turn_context), runtime, &request, input).await;
}

pub(crate) fn start_startup_tasks(sess: &Arc<Session>) {
    let sess = Arc::clone(sess);
    tokio::spawn(async move {
        if let Err(err) = run_startup_tasks(sess).await {
            tracing::warn!("failed to run LCM durable-memory startup maintenance: {err}");
        }
    });
}

async fn run_startup_tasks(sess: Arc<Session>) -> CodexResult<()> {
    let Some(runtime) = sess.open_brain_runtime().await else {
        return Ok(());
    };
    let turn_context = sess.new_default_turn().await;
    let request = RequestContext::from_turn_context(turn_context.as_ref());

    backfill_latest_summary(sess.as_ref(), &runtime, &request).await?;
    upgrade_recent_project_memories(sess.as_ref(), &runtime, &request).await?;
    Ok(())
}

async fn backfill_latest_summary(
    sess: &Session,
    runtime: &OpenBrainRuntime,
    request: &RequestContext,
) -> CodexResult<()> {
    let promotion_mode = memory_promotion_mode(sess, sess.conversation_id).await;
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

    let input = DurableMemoryExtractionInput {
        source_summary_node_id: Some(candidate.summary_node_id.clone()),
        latest_user_message: normalized_candidate_title(&candidate.title),
        summary_text: candidate.content,
        evidence_text: None,
        source_node_ids: vec![candidate.summary_node_id.clone()],
        source_event_ids: candidate.source_event_ids,
        source_token_count: candidate.source_token_count,
        summary_token_count: candidate.summary_token_count,
        promotion_source: "startup backfill",
        supersede_node_ids: Vec::new(),
        mirror_to_thoughts: runtime.mirror_durable_to_thoughts(),
    };
    promote_records(sess, None, runtime, request, input).await;
    Ok(())
}

async fn upgrade_recent_project_memories(
    sess: &Session,
    runtime: &OpenBrainRuntime,
    request: &RequestContext,
) -> CodexResult<()> {
    let promotion_mode = memory_promotion_mode(sess, sess.conversation_id).await;
    if !promotion_mode.allows_promotion() {
        tracing::debug!(
            "skipping LCM durable-memory upgrade: thread memory mode disallows promotion"
        );
        return Ok(());
    }

    let memories = runtime
        .list_project_durable_memories(32)
        .await
        .map_err(|err| {
            CodexErr::Fatal(format!("Open Brain durable-memory upgrade failed: {err}"))
        })?;
    let mut upgraded = 0usize;
    for memory in memories
        .iter()
        .filter(|memory| memory_needs_upgrade(memory))
    {
        if upgraded >= MAX_UPGRADE_BATCH {
            break;
        }
        let Some(input) = upgrade_input(memory, runtime.mirror_durable_to_thoughts()) else {
            continue;
        };
        promote_records(sess, None, runtime, request, input).await;
        upgraded += 1;
    }
    if upgraded > 0 {
        info!("ran LCM durable-memory upgrade for {upgraded} recent project memory record(s)");
    }
    Ok(())
}

async fn promote_records(
    sess: &Session,
    turn_context: Option<&TurnContext>,
    runtime: &OpenBrainRuntime,
    request: &RequestContext,
    input: DurableMemoryExtractionInput,
) {
    if input.summary_text.trim().is_empty() {
        tracing::debug!(
            promotion_source = input.promotion_source,
            "skipping LCM durable-memory promotion: empty summary text"
        );
        return;
    }

    let records = match extract_records(sess, request, &input).await {
        Ok(records) => records,
        Err(err) => {
            warn!(
                promotion_source = input.promotion_source,
                "LCM durable-memory extraction failed: {err}"
            );
            if let Some(turn_context) = turn_context {
                sess.notify_background_event(
                    turn_context,
                    format!("LCM durable-memory extraction failed: {err}"),
                )
                .await;
            }
            return;
        }
    };

    if records.is_empty() {
        tracing::debug!(
            promotion_source = input.promotion_source,
            "LCM durable-memory extraction produced no promotable records"
        );
        return;
    }

    match runtime.promote_durable_memory(&records).await {
        Ok(result) => {
            tracing::info!(
                inserted_count = result.inserted_count,
                promotion_source = input.promotion_source,
                "Open Brain durable-memory promotion succeeded"
            );
            if result.inserted_count > 0
                && let Some(turn_context) = turn_context
            {
                let promoted = records
                    .iter()
                    .map(|record| format!("{}: {}", record.memory_type, record.title))
                    .collect::<Vec<_>>()
                    .join("; ");
                sess.notify_background_event(
                    turn_context,
                    format!("LCM durable memory promoted: {promoted}."),
                )
                .await;
            }
        }
        Err(err) => {
            warn!(
                promotion_source = input.promotion_source,
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

async fn extract_records(
    sess: &Session,
    request: &RequestContext,
    input: &DurableMemoryExtractionInput,
) -> anyhow::Result<Vec<OpenBrainDurableMemoryRecord>> {
    let prompt = Prompt {
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: build_extraction_input_message(input),
            }],
            end_turn: None,
            phase: None,
        }],
        tools: Vec::new(),
        parallel_tool_calls: false,
        base_instructions: BaseInstructions {
            text: EXTRACTION_PROMPT.to_string(),
        },
        personality: None,
        output_schema: Some(output_schema()),
    };

    let mut client_session = sess.services.model_client.new_session();
    let mut stream = client_session
        .stream(
            &prompt,
            &request.model_info,
            &request.session_telemetry,
            request.reasoning_effort,
            request.reasoning_summary,
            request.service_tier,
            request.turn_metadata_header.as_deref(),
        )
        .await?;

    let mut result = String::new();
    while let Some(message) = stream.next().await.transpose()? {
        match message {
            ResponseEvent::OutputTextDelta(delta) => result.push_str(&delta),
            ResponseEvent::OutputItemDone(item) => {
                if result.is_empty()
                    && let ResponseItem::Message { content, .. } = item
                    && let Some(text) = content_items_to_text(&content)
                {
                    result.push_str(&text);
                }
            }
            ResponseEvent::Completed { .. } => break,
            _ => {}
        }
    }

    let output: DurableMemoryModelOutput = serde_json::from_str(&result)?;
    Ok(output
        .records
        .iter()
        .filter_map(|record| promoteable_record(input, record))
        .take(MAX_PROMOTED_RECORDS)
        .collect())
}

fn output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "maxItems": MAX_PROMOTED_RECORDS,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": [
                                "decision",
                                "constraint",
                                "known_issue",
                                "implementation_status",
                                "operator_preference"
                            ]
                        },
                        "title": { "type": "string" },
                        "content": { "type": "string" },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": ["memory_type", "title", "content", "confidence"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["records"],
        "additionalProperties": false
    })
}

fn build_extraction_input_message(input: &DurableMemoryExtractionInput) -> String {
    let mut lines = vec![format!("promotion_source = {}", input.promotion_source)];
    if let Some(summary_node_id) = input.source_summary_node_id.as_deref() {
        lines.push(format!("source_summary_node_id = {summary_node_id}"));
    }
    if let Some(latest_user_message) = input.latest_user_message.as_deref() {
        lines.push(format!("latest_user_message = {latest_user_message}"));
    }
    if let Some(source_tokens) = input.source_token_count {
        lines.push(format!("source_token_count = {source_tokens}"));
    }
    if let Some(summary_tokens) = input.summary_token_count {
        lines.push(format!("summary_token_count = {summary_tokens}"));
    }
    lines.push(String::new());
    lines.push("summary:".to_string());
    lines.push(input.summary_text.trim().to_string());
    if let Some(evidence_text) = input.evidence_text.as_deref() {
        lines.push(String::new());
        lines.push("evidence:".to_string());
        lines.push(evidence_text.to_string());
    }
    lines.join("\n")
}

fn promoteable_record(
    input: &DurableMemoryExtractionInput,
    record: &DurableMemoryModelRecord,
) -> Option<OpenBrainDurableMemoryRecord> {
    if record.confidence < CONFIDENCE_THRESHOLD {
        return None;
    }

    let title = redact_secrets(normalize_title(&record.title)?);
    let content = redact_secrets(normalize_content(&record.content)?);
    let memory_type = normalize_memory_type(&record.memory_type)?;
    let memory_key = compute_memory_key(&memory_type, &title);

    let mut metadata = BTreeMap::from([
        (
            "quality_version".to_string(),
            serde_json::Value::from(DURABLE_MEMORY_QUALITY_VERSION),
        ),
        (
            "confidence".to_string(),
            serde_json::Value::from(record.confidence),
        ),
        (
            "promotion_source".to_string(),
            serde_json::Value::String(input.promotion_source.to_string()),
        ),
    ]);
    if let Some(summary_node_id) = input.source_summary_node_id.as_deref() {
        metadata.insert(
            "source_summary_node_id".to_string(),
            serde_json::Value::String(summary_node_id.to_string()),
        );
    }
    if let Some(source_token_count) = input.source_token_count {
        metadata.insert(
            "source_token_count".to_string(),
            serde_json::Value::from(source_token_count),
        );
    }
    if let Some(summary_token_count) = input.summary_token_count {
        metadata.insert(
            "summary_token_count".to_string(),
            serde_json::Value::from(summary_token_count),
        );
    }

    Some(OpenBrainDurableMemoryRecord {
        title,
        content,
        memory_type,
        memory_key: Some(memory_key),
        source_node_ids: input.source_node_ids.clone(),
        source_event_ids: input.source_event_ids.clone(),
        supersede_node_ids: input.supersede_node_ids.clone(),
        metadata,
        mirror_to_thoughts: input.mirror_to_thoughts,
    })
}

fn normalize_memory_type(value: &str) -> Option<String> {
    match value.trim() {
        "decision"
        | "constraint"
        | "known_issue"
        | "implementation_status"
        | "operator_preference" => Some(value.trim().to_string()),
        _ => None,
    }
}

fn normalize_title(title: &str) -> Option<String> {
    let mut normalized = title
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '-' || c == ':');
    if normalized.is_empty() {
        return None;
    }
    let lowered = normalized.to_ascii_lowercase();
    for prefix in [
        "lcm memory:",
        "memory:",
        "issue:",
        "decision:",
        "constraint:",
    ] {
        if lowered.starts_with(prefix) {
            normalized = normalized[prefix.len()..].trim();
            break;
        }
    }
    if normalized.is_empty() {
        return None;
    }
    if normalized.ends_with('?') || is_generic_title(normalized) {
        return None;
    }
    Some(preview_text(normalized, 96))
}

fn normalize_content(content: &str) -> Option<String> {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() || looks_like_polluted_memory(&normalized) {
        return None;
    }
    Some(preview_text(&normalized, 280))
}

fn is_generic_title(title: &str) -> bool {
    matches!(
        title.trim().to_ascii_lowercase().as_str(),
        "durable memory"
            | "project durable memory"
            | "lcm durable memory"
            | "summary"
            | "recap"
            | "note"
    )
}

fn looks_like_polluted_memory(content: &str) -> bool {
    let lowered = content.to_ascii_lowercase();
    lowered.starts_with("assistant:")
        || lowered.starts_with("user:")
        || lowered.contains("<context_engine>")
        || lowered.contains("<permissions instructions>")
        || lowered.contains("# agents.md instructions")
}

fn evidence_text(items: &[ResponseItem]) -> Option<String> {
    let rendered = items
        .iter()
        .rev()
        .filter_map(render_history_item)
        .take(8)
        .collect::<Vec<_>>()
        .join("\n");
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(codex_utils_output_truncation::truncate_text(
        trimmed,
        codex_utils_output_truncation::TruncationPolicy::Tokens(EXTRACTION_EVIDENCE_TOKEN_BUDGET),
    ))
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

fn latest_user_message_text(items: &[ResponseItem]) -> Option<String> {
    items.iter().rev().find_map(|item| match item {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content)
        }
        _ => None,
    })
}

fn normalized_candidate_title(title: &str) -> Option<String> {
    title
        .strip_prefix("LCM memory: ")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn upgrade_input(
    memory: &OpenBrainProjectMemory,
    mirror_to_thoughts: bool,
) -> Option<DurableMemoryExtractionInput> {
    let summary_text = memory.content.trim();
    if summary_text.is_empty() {
        return None;
    }
    let source_summary_node_id = memory.source_node_ids.first().cloned();
    Some(DurableMemoryExtractionInput {
        source_summary_node_id,
        latest_user_message: None,
        summary_text: summary_text.to_string(),
        evidence_text: None,
        source_node_ids: memory.source_node_ids.clone(),
        source_event_ids: memory.source_event_ids.clone(),
        source_token_count: None,
        summary_token_count: None,
        promotion_source: "recent quality upgrade",
        supersede_node_ids: vec![memory.node_id.clone()],
        mirror_to_thoughts,
    })
}

fn memory_needs_upgrade(memory: &OpenBrainProjectMemory) -> bool {
    if memory.quality_version.unwrap_or_default() < DURABLE_MEMORY_QUALITY_VERSION {
        return true;
    }
    memory.memory_type.as_deref().map_or(true, |memory_type| {
        memory_type == "lcm_leaf_summary" || memory_type == "durable_memory"
    }) || is_generic_title(&memory.title)
        || looks_like_polluted_memory(memory.content.trim())
}

fn compute_memory_key(memory_type: &str, title: &str) -> String {
    let canonical_title = title
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut digest = Sha256::new();
    digest.update(format!("durable_memory|{memory_type}|{canonical_title}"));
    format!("{:x}", digest.finalize())
}

async fn memory_promotion_mode(sess: &Session, thread_id: ThreadId) -> LcmMemoryPromotionMode {
    let Some(state_db) = sess.services.state_db.as_deref() else {
        return LcmMemoryPromotionMode::Enabled;
    };
    match state_db.get_thread_memory_mode(thread_id).await {
        Ok(mode) => resolve_memory_promotion_mode(mode.as_deref()),
        Err(err) => {
            tracing::warn!("failed to read thread memory mode for LCM promotion: {err}");
            LcmMemoryPromotionMode::Enabled
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

fn resolve_memory_promotion_mode(mode: Option<&str>) -> LcmMemoryPromotionMode {
    match mode {
        Some("disabled") => LcmMemoryPromotionMode::Disabled,
        Some("enabled") | None => LcmMemoryPromotionMode::Enabled,
        Some(legacy) => LcmMemoryPromotionMode::Legacy(legacy.to_string()),
    }
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
    use super::compute_memory_key;
    use super::is_generic_title;
    use super::looks_like_polluted_memory;
    use super::memory_needs_upgrade;
    use super::normalize_content;
    use super::normalize_title;
    use codex_open_brain::OpenBrainProjectMemory;
    use pretty_assertions::assert_eq;

    #[test]
    fn normalize_title_rejects_generic_titles() {
        assert_eq!(normalize_title("LCM durable memory"), None);
        assert_eq!(normalize_title("What changed?"), None);
    }

    #[test]
    fn normalize_content_rejects_polluted_bodies() {
        assert_eq!(normalize_content("assistant: raw transcript"), None);
        assert!(looks_like_polluted_memory(
            "<context_engine> engine = open_brain_lcm"
        ));
    }

    #[test]
    fn compute_memory_key_is_stable_for_title_whitespace() {
        assert_eq!(
            compute_memory_key("known_issue", "Compaction integrity bug"),
            compute_memory_key("known_issue", "  compaction   integrity bug  ")
        );
    }

    #[test]
    fn memory_needs_upgrade_flags_generic_memories() {
        let memory = OpenBrainProjectMemory {
            node_id: "node-1".to_string(),
            node_kind: "durable_memory".to_string(),
            title: "LCM durable memory".to_string(),
            content: "assistant: bad".to_string(),
            memory_key: None,
            memory_type: Some("durable_memory".to_string()),
            quality_version: None,
            project_key: Some("/tmp/project".to_string()),
            scope_key: Some("/tmp/project".to_string()),
            updated_at: None,
            source_thought_id: None,
            source_node_ids: vec!["summary-1".to_string()],
            source_event_ids: vec!["event-1".to_string()],
            superseded_at: None,
        };

        assert!(memory_needs_upgrade(&memory));
        assert!(is_generic_title(&memory.title));
    }
}
