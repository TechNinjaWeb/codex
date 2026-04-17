use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use reqwest::header::AUTHORIZATION;
use reqwest::header::CONTENT_TYPE;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::header::InvalidHeaderValue;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn bool_true() -> bool {
    true
}

fn default_fresh_tail_count() -> usize {
    64
}

fn default_context_threshold() -> f64 {
    0.75
}

fn default_leaf_chunk_tokens() -> usize {
    20_000
}

fn default_leaf_target_tokens() -> usize {
    1_200
}

fn default_condensed_target_tokens() -> usize {
    2_000
}

fn default_condensed_min_fanout() -> usize {
    4
}

fn default_incremental_max_depth() -> usize {
    1
}

fn default_max_expand_tokens() -> usize {
    4_000
}

fn default_new_session_retain_depth() -> usize {
    2
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpenBrainProjectScopeStrategy {
    #[default]
    GitRootFirst,
    WorkspaceRoot,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct OpenBrainConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supabase_url_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_role_key_env: Option<String>,
    #[serde(default)]
    pub project_scope_strategy: OpenBrainProjectScopeStrategy,
    #[serde(default = "bool_true")]
    pub mirror_durable_to_thoughts: bool,
}

impl Default for OpenBrainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            supabase_url_env: None,
            service_role_key_env: None,
            project_scope_strategy: OpenBrainProjectScopeStrategy::default(),
            mirror_durable_to_thoughts: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct LcmConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default = "default_fresh_tail_count")]
    pub fresh_tail_count: usize,
    #[serde(default = "default_context_threshold")]
    pub context_threshold: f64,
    #[serde(default = "default_leaf_chunk_tokens")]
    pub leaf_chunk_tokens: usize,
    #[serde(default = "default_leaf_target_tokens")]
    pub leaf_target_tokens: usize,
    #[serde(default = "default_condensed_target_tokens")]
    pub condensed_target_tokens: usize,
    #[serde(default = "default_condensed_min_fanout")]
    pub condensed_min_fanout: usize,
    #[serde(default = "default_incremental_max_depth")]
    pub incremental_max_depth: usize,
    #[serde(default = "default_max_expand_tokens")]
    pub max_expand_tokens: usize,
    #[serde(default = "default_new_session_retain_depth")]
    pub new_session_retain_depth: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expansion_model: Option<String>,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fresh_tail_count: default_fresh_tail_count(),
            context_threshold: default_context_threshold(),
            leaf_chunk_tokens: default_leaf_chunk_tokens(),
            leaf_target_tokens: default_leaf_target_tokens(),
            condensed_target_tokens: default_condensed_target_tokens(),
            condensed_min_fanout: default_condensed_min_fanout(),
            incremental_max_depth: default_incremental_max_depth(),
            max_expand_tokens: default_max_expand_tokens(),
            new_session_retain_depth: default_new_session_retain_depth(),
            summary_model: None,
            expansion_model: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainStoreConfig {
    pub supabase_url: String,
    pub service_role_key: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextPacketItemKind {
    RecentTail,
    GraphFrontier,
    DurableMemory,
    ExplicitExpansion,
    WorkingMemory,
    SystemInstruction,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OpenBrainRelationship {
    Summarizes,
    Condenses,
    ExpandsTo,
    DerivesFrom,
    Supersedes,
    PromotesTo,
    BelongsToProject,
    ForkedFrom,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainArtifactRef {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationship: Option<OpenBrainRelationship>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ContextPacketItem {
    pub kind: ContextPacketItemKind,
    pub content: String,
    #[serde(default)]
    pub source_refs: Vec<OpenBrainArtifactRef>,
    pub selection_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_tok: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desc_tok: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ContextPacket {
    pub packet_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_key: Option<String>,
    pub token_budget: usize,
    #[serde(default)]
    pub items: Vec<ContextPacketItem>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainSyncRecord {
    pub role: String,
    pub item_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linked_tool_call_id: Option<String>,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub payload: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_index: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenBrainNodeKind {
    RawTurn,
    SummaryD0,
    SummaryD1Plus,
    DurableMemory,
    ContextPacket,
    ExpansionResult,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainNodeRecord {
    pub node_kind: OpenBrainNodeKind,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_tok: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desc_tok: Option<u32>,
    #[serde(default)]
    pub source_item_ids: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainAppendResult {
    pub session_id: String,
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_key: Option<String>,
    pub inserted_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_sequence: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainLeafSummary {
    pub summary_node_id: String,
    pub thread_id: String,
    pub depth: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainGraphStatus {
    #[serde(default)]
    pub fresh_tail_count: usize,
    #[serde(default)]
    pub packet_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_packet_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_packet_token_budget: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_packet_query: Option<String>,
    #[serde(default)]
    pub node_kind_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub depth_counts: BTreeMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainGraphView {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_key: Option<String>,
    #[serde(default)]
    pub nodes: Vec<JsonValue>,
    #[serde(default)]
    pub edges: Vec<JsonValue>,
    #[serde(default)]
    pub fresh_tail: Vec<JsonValue>,
    #[serde(default)]
    pub context_packets: Vec<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OpenBrainGraphStatus>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainNodeDescription {
    pub node: JsonValue,
    #[serde(default)]
    pub linked_thoughts: Vec<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<JsonValue>,
    #[serde(default)]
    pub context_packets: Vec<JsonValue>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainDurableMemoryCandidate {
    pub summary_node_id: String,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub source_event_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_token_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_token_count: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainSearchResults {
    pub query: String,
    #[serde(default)]
    pub results: Vec<JsonValue>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainExpansion {
    pub node: JsonValue,
    #[serde(default)]
    pub incoming: Vec<JsonValue>,
    #[serde(default)]
    pub outgoing: Vec<JsonValue>,
    #[serde(default)]
    pub events: Vec<JsonValue>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenBrainExpandQueryResults {
    pub thread_id: String,
    pub query: String,
    pub token_budget: usize,
    #[serde(default)]
    pub results: Vec<OpenBrainExpansion>,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenBrainClientError {
    #[error("invalid Open Brain header value: {0}")]
    InvalidHeader(#[from] InvalidHeaderValue),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("unexpected Open Brain status {status}: {body}")]
    UnexpectedStatus {
        status: reqwest::StatusCode,
        body: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum OpenBrainRuntimeError {
    #[error("Open Brain is disabled")]
    Disabled,
    #[error("missing Open Brain env var `{0}`")]
    MissingEnvVar(String),
    #[error("Open Brain client error: {0}")]
    Client(#[from] OpenBrainClientError),
}

#[derive(Clone, Debug)]
pub struct OpenBrainRpcClient {
    client: reqwest::Client,
    config: OpenBrainStoreConfig,
}

impl OpenBrainRpcClient {
    #[must_use]
    pub fn new(config: OpenBrainStoreConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    pub async fn rpc_json<T, P>(&self, name: &str, payload: &P) -> Result<T, OpenBrainClientError>
    where
        T: DeserializeOwned,
        P: Serialize + ?Sized,
    {
        let response = self
            .client
            .post(format!(
                "{}/rest/v1/rpc/{name}",
                self.config.supabase_url.trim_end_matches('/')
            ))
            .headers(self.headers()?)
            .json(payload)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(OpenBrainClientError::UnexpectedStatus {
                status,
                body: response.text().await.unwrap_or_default(),
            });
        }
        Ok(response.json().await?)
    }

    pub async fn rest_get_json<T>(
        &self,
        table: &str,
        query: &[(&str, String)],
    ) -> Result<T, OpenBrainClientError>
    where
        T: DeserializeOwned,
    {
        let response = self
            .client
            .get(format!(
                "{}/rest/v1/{table}",
                self.config.supabase_url.trim_end_matches('/')
            ))
            .headers(self.headers()?)
            .query(query)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(OpenBrainClientError::UnexpectedStatus {
                status,
                body: response.text().await.unwrap_or_default(),
            });
        }
        Ok(response.json().await?)
    }

    fn headers(&self) -> Result<HeaderMap, OpenBrainClientError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "apikey",
            HeaderValue::from_str(self.config.service_role_key.as_str())?,
        );
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(format!("Bearer {}", self.config.service_role_key).as_str())?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }
}

#[derive(Clone, Debug)]
pub struct OpenBrainRuntime {
    client: OpenBrainRpcClient,
    lcm: LcmConfig,
    mirror_durable_to_thoughts: bool,
    session_id: String,
    thread_id: String,
    rollout_id: String,
    project_key: String,
    scope_key: String,
}

impl OpenBrainRuntime {
    pub fn from_env(
        config: &OpenBrainConfig,
        lcm: &LcmConfig,
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
        rollout_id: impl Into<String>,
        cwd: &Path,
    ) -> Result<Self, OpenBrainRuntimeError> {
        if !config.enabled {
            return Err(OpenBrainRuntimeError::Disabled);
        }
        let supabase_url_env = config
            .supabase_url_env
            .clone()
            .unwrap_or_else(|| "SUPABASE_URL".to_string());
        let service_role_key_env = config
            .service_role_key_env
            .clone()
            .unwrap_or_else(|| "SUPABASE_SERVICE_ROLE_KEY".to_string());
        let supabase_url = std::env::var(&supabase_url_env)
            .map_err(|_| OpenBrainRuntimeError::MissingEnvVar(supabase_url_env.clone()))?;
        let service_role_key = std::env::var(&service_role_key_env)
            .map_err(|_| OpenBrainRuntimeError::MissingEnvVar(service_role_key_env.clone()))?;
        let session_id = session_id.into();
        let project_key = derive_project_key(cwd, config.project_scope_strategy);
        Ok(Self {
            client: OpenBrainRpcClient::new(OpenBrainStoreConfig {
                supabase_url,
                service_role_key,
            }),
            lcm: lcm.clone(),
            mirror_durable_to_thoughts: config.mirror_durable_to_thoughts,
            thread_id: thread_id.into(),
            rollout_id: rollout_id.into(),
            scope_key: project_key.clone(),
            project_key,
            session_id,
        })
    }

    pub fn session_id(&self) -> &str {
        self.session_id.as_str()
    }

    pub fn thread_id(&self) -> &str {
        self.thread_id.as_str()
    }

    pub fn project_key(&self) -> &str {
        self.project_key.as_str()
    }

    pub fn scope_key(&self) -> &str {
        self.scope_key.as_str()
    }

    pub fn lcm(&self) -> &LcmConfig {
        &self.lcm
    }

    pub fn mirror_durable_to_thoughts(&self) -> bool {
        self.mirror_durable_to_thoughts
    }

    pub fn context_token_budget(&self, model_context_window: Option<i64>) -> usize {
        let contextual_budget = model_context_window
            .filter(|window| *window > 0)
            .map(|window| {
                ((window as f64) * (1.0 - self.lcm.context_threshold))
                    .round()
                    .max(256.0) as usize
            })
            .unwrap_or(self.lcm.max_expand_tokens);
        contextual_budget.min(self.lcm.max_expand_tokens).max(256)
    }

    pub async fn append_response_items(
        &self,
        turn_id: &str,
        items: &[ResponseItem],
    ) -> Result<OpenBrainAppendResult, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_session_id": self.session_id,
            "p_thread_id": self.thread_id,
            "p_rollout_id": self.rollout_id,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
            "p_items": self.to_sync_records(turn_id, items)?,
        });
        self.client
            .rpc_json("ob_append_rollout_items", &payload)
            .await
    }

    pub async fn compact_leaf(
        &self,
        content: &str,
        source_event_refs: &[String],
        src_tok: u32,
        desc_tok: u32,
    ) -> Result<OpenBrainLeafSummary, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_session_id": self.session_id,
            "p_thread_id": self.thread_id,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
            "p_content": content,
            "p_source_event_ids": source_event_refs,
            "p_title": "LCM leaf summary",
            "p_depth": 0,
            "p_src_tok": src_tok,
            "p_desc_tok": desc_tok,
            "p_metadata": {
                "leaf_target_tokens": self.lcm.leaf_target_tokens,
                "fresh_tail_count": self.lcm.fresh_tail_count,
            },
            "p_provenance": {
                "source": "codex-open-brain",
                "rollout_id": self.rollout_id,
            },
        });
        self.client.rpc_json("ob_compact_leaf", &payload).await
    }

    pub fn source_event_refs_for_items(&self, items: &[ResponseItem]) -> Vec<String> {
        items
            .iter()
            .enumerate()
            .map(|(index, item)| response_item_id(item, index))
            .collect()
    }

    pub async fn assemble_context(
        &self,
        query: Option<&str>,
        token_budget: usize,
    ) -> Result<ContextPacket, OpenBrainClientError> {
        let payload = serde_json::json!({
            "session_id": self.session_id,
            "query": query.unwrap_or_default(),
            "token_budget": token_budget,
            "p_thread_id": self.thread_id,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
        });
        self.client.rpc_json("ob_assemble_context", &payload).await
    }

    pub async fn graph_view(
        &self,
        include_superseded: bool,
        limit: usize,
    ) -> Result<OpenBrainGraphView, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_thread_id": self.thread_id,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
            "p_limit": limit.max(1),
            "p_include_superseded": include_superseded,
        });
        self.client.rpc_json("ob_graph_view", &payload).await
    }

    pub async fn describe_node(
        &self,
        node_id: &str,
        include_superseded: bool,
    ) -> Result<OpenBrainNodeDescription, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_node_id": node_id,
            "p_include_superseded": include_superseded,
        });
        self.client.rpc_json("ob_describe_node", &payload).await
    }

    pub async fn search(
        &self,
        query: &str,
        include_superseded: bool,
        limit: usize,
    ) -> Result<OpenBrainSearchResults, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_query": query,
            "p_thread_id": self.thread_id,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
            "p_limit": limit.max(1),
            "p_include_superseded": include_superseded,
        });
        self.client.rpc_json("ob_grep", &payload).await
    }

    pub async fn expand(
        &self,
        node_id: &str,
        limit: usize,
    ) -> Result<OpenBrainExpansion, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_node_id": node_id,
            "p_limit": limit.max(1),
        });
        self.client.rpc_json("ob_expand", &payload).await
    }

    pub async fn expand_query(
        &self,
        query: &str,
        limit: usize,
        token_budget: usize,
    ) -> Result<OpenBrainExpandQueryResults, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_thread_id": self.thread_id,
            "p_query": query,
            "p_project_key": self.project_key,
            "p_scope_key": self.scope_key,
            "p_limit": limit.max(1),
            "p_token_budget": token_budget.max(256),
        });
        self.client.rpc_json("ob_expand_query", &payload).await
    }

    pub async fn list_context_packets(
        &self,
        limit: usize,
    ) -> Result<Vec<JsonValue>, OpenBrainClientError> {
        let graph = self
            .graph_view(/*include_superseded*/ false, limit.max(16))
            .await?;
        Ok(graph
            .context_packets
            .into_iter()
            .take(limit.max(1))
            .collect())
    }

    pub async fn latest_unpromoted_summary_candidate(
        &self,
        limit: usize,
    ) -> Result<Option<OpenBrainDurableMemoryCandidate>, OpenBrainClientError> {
        let graph = self
            .graph_view(/*include_superseded*/ false, limit.max(64))
            .await?;
        Ok(latest_unpromoted_summary_candidate_from_graph(&graph))
    }

    pub async fn promote_durable_memory(
        &self,
        records: &[OpenBrainDurableMemoryRecord],
    ) -> Result<OpenBrainPromotionResult, OpenBrainClientError> {
        let payload = serde_json::json!({
            "p_session_id": self.session_id,
            "p_records": records,
        });
        self.client
            .rpc_json("ob_promote_durable_memory", &payload)
            .await
    }

    fn to_sync_records(
        &self,
        turn_id: &str,
        items: &[ResponseItem],
    ) -> Result<Vec<OpenBrainSyncRecord>, OpenBrainClientError> {
        let timestamp_ms = now_unix_millis();
        items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let (item_kind, role, tool_call_id, linked_tool_call_id) =
                    response_item_shape(item);
                let content = response_item_text(item);
                let mut payload = json_object(item)?;
                if let Some(text) = content.as_ref() {
                    payload.insert("text".to_string(), serde_json::Value::String(text.clone()));
                }
                Ok(OpenBrainSyncRecord {
                    role: role.to_string(),
                    item_id: response_item_id(item, index),
                    turn_id: Some(turn_id.to_string()),
                    tool_call_id,
                    linked_tool_call_id,
                    timestamp_ms,
                    payload,
                    item_kind: Some(item_kind.to_string()),
                    content,
                    item_index: Some(index),
                })
            })
            .collect()
    }
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn derive_project_key(cwd: &Path, strategy: OpenBrainProjectScopeStrategy) -> String {
    match strategy {
        OpenBrainProjectScopeStrategy::WorkspaceRoot => cwd.to_string_lossy().into_owned(),
        OpenBrainProjectScopeStrategy::GitRootFirst => discover_git_root(cwd)
            .unwrap_or(cwd)
            .to_string_lossy()
            .into_owned(),
    }
}

fn discover_git_root(start: &Path) -> Option<&Path> {
    start.ancestors().find(|path| path.join(".git").exists())
}

fn graph_json_str<'a>(value: &'a JsonValue, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(JsonValue::as_str))
}

fn graph_json_u32(value: &JsonValue, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(JsonValue::as_u64))
        .and_then(|value| u32::try_from(value).ok())
}

fn graph_json_string_array(value: &JsonValue, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| {
            value.get(*key).and_then(JsonValue::as_array).map(|items| {
                items
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
}

fn latest_unpromoted_summary_candidate_from_graph(
    graph: &OpenBrainGraphView,
) -> Option<OpenBrainDurableMemoryCandidate> {
    let promoted_targets = graph
        .edges
        .iter()
        .filter(|edge| {
            graph_json_str(edge, &["relationship_type", "relationshipType"]) == Some("PROMOTES_TO")
        })
        .filter_map(|edge| graph_json_str(edge, &["to_node_id", "toNodeId"]).map(str::to_string))
        .collect::<BTreeSet<_>>();

    let latest_packet_query = graph
        .status
        .as_ref()
        .and_then(|status| status.latest_packet_query.as_ref())
        .map(|query| query.trim())
        .filter(|query| !query.is_empty())
        .map(ToString::to_string);

    graph.nodes.iter().find_map(|node| {
        let node_kind = graph_json_str(node, &["node_kind", "nodeKind"])?;
        if !matches!(node_kind, "summary_d0" | "summary_d1plus") {
            return None;
        }
        let node_id = graph_json_str(node, &["node_id", "id"])?;
        if promoted_targets.contains(node_id) {
            return None;
        }
        let content = graph_json_str(node, &["content"])?.trim().to_string();
        if content.is_empty() {
            return None;
        }

        let title = latest_packet_query
            .clone()
            .or_else(|| {
                graph_json_str(node, &["title"])
                    .map(str::trim)
                    .filter(|title| !title.is_empty())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "LCM durable memory".to_string());

        Some(OpenBrainDurableMemoryCandidate {
            summary_node_id: node_id.to_string(),
            title,
            content,
            source_event_ids: graph_json_string_array(node, &["source_event_ids"]),
            source_token_count: graph_json_u32(node, &["src_tok", "srcTok"]),
            summary_token_count: graph_json_u32(node, &["desc_tok", "descTok"]),
        })
    })
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainDurableMemoryRecord {
    pub title: String,
    pub content: String,
    pub memory_type: String,
    #[serde(default)]
    pub source_node_ids: Vec<String>,
    #[serde(default)]
    pub source_event_ids: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub mirror_to_thoughts: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OpenBrainPromotionResult {
    pub session_id: String,
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
    pub inserted_count: usize,
}

fn json_object(
    item: &ResponseItem,
) -> Result<BTreeMap<String, serde_json::Value>, OpenBrainClientError> {
    let value = serde_json::to_value(item)?;
    Ok(match value {
        serde_json::Value::Object(map) => map.into_iter().collect(),
        other => BTreeMap::from([("item".to_string(), other)]),
    })
}

fn response_item_id(item: &ResponseItem, index: usize) -> String {
    match item {
        ResponseItem::Message { id, .. } => {
            id.clone().unwrap_or_else(|| format!("message-{index}"))
        }
        ResponseItem::Reasoning { id, .. } => id.clone(),
        ResponseItem::LocalShellCall { call_id, .. } => call_id
            .clone()
            .unwrap_or_else(|| format!("local-shell-{index}")),
        ResponseItem::FunctionCall { call_id, .. } => call_id.clone(),
        ResponseItem::FunctionCallOutput { call_id, .. } => format!("function-output-{call_id}"),
        ResponseItem::CustomToolCall { call_id, .. } => call_id.clone(),
        ResponseItem::CustomToolCallOutput { call_id, .. } => {
            format!("custom-tool-output-{call_id}")
        }
        ResponseItem::ToolSearchCall { call_id, .. } => call_id
            .clone()
            .unwrap_or_else(|| format!("tool-search-{index}")),
        ResponseItem::ToolSearchOutput { call_id, .. } => call_id
            .clone()
            .map(|call_id| format!("tool-search-output-{call_id}"))
            .unwrap_or_else(|| format!("tool-search-output-{index}")),
        ResponseItem::WebSearchCall { id, .. } => {
            id.clone().unwrap_or_else(|| format!("web-search-{index}"))
        }
        ResponseItem::ImageGenerationCall { id, .. } => id.clone(),
        ResponseItem::GhostSnapshot { .. } => format!("ghost-snapshot-{index}"),
        ResponseItem::Compaction { .. } => format!("compaction-{index}"),
        ResponseItem::Other => format!("other-{index}"),
    }
}

fn response_item_shape(item: &ResponseItem) -> (String, String, Option<String>, Option<String>) {
    match item {
        ResponseItem::Message { role, .. } => ("message".to_string(), role.clone(), None, None),
        ResponseItem::Reasoning { .. } => {
            ("reasoning".to_string(), "assistant".to_string(), None, None)
        }
        ResponseItem::LocalShellCall { call_id, .. } => (
            "local_shell_call".to_string(),
            "tool".to_string(),
            call_id.clone(),
            None,
        ),
        ResponseItem::FunctionCall { call_id, .. } => (
            "function_call".to_string(),
            "assistant".to_string(),
            Some(call_id.clone()),
            None,
        ),
        ResponseItem::FunctionCallOutput { call_id, .. } => (
            "function_call_output".to_string(),
            "tool".to_string(),
            None,
            Some(call_id.clone()),
        ),
        ResponseItem::CustomToolCall { call_id, .. } => (
            "custom_tool_call".to_string(),
            "assistant".to_string(),
            Some(call_id.clone()),
            None,
        ),
        ResponseItem::CustomToolCallOutput { call_id, .. } => (
            "custom_tool_call_output".to_string(),
            "tool".to_string(),
            None,
            Some(call_id.clone()),
        ),
        ResponseItem::ToolSearchCall { call_id, .. } => (
            "tool_search_call".to_string(),
            "assistant".to_string(),
            call_id.clone(),
            None,
        ),
        ResponseItem::ToolSearchOutput { call_id, .. } => (
            "tool_search_output".to_string(),
            "tool".to_string(),
            None,
            call_id.clone(),
        ),
        ResponseItem::WebSearchCall { .. } => (
            "web_search_call".to_string(),
            "assistant".to_string(),
            None,
            None,
        ),
        ResponseItem::ImageGenerationCall { .. } => (
            "image_generation_call".to_string(),
            "assistant".to_string(),
            None,
            None,
        ),
        ResponseItem::GhostSnapshot { .. } => (
            "ghost_snapshot".to_string(),
            "system".to_string(),
            None,
            None,
        ),
        ResponseItem::Compaction { .. } => {
            ("compaction".to_string(), "system".to_string(), None, None)
        }
        ResponseItem::Other => ("other".to_string(), "unknown".to_string(), None, None),
    }
}

fn response_item_text(item: &ResponseItem) -> Option<String> {
    match item {
        ResponseItem::Message { content, .. } => content_items_to_text(content),
        ResponseItem::Reasoning { summary, .. } => {
            let joined = summary
                .iter()
                .filter_map(|item| match item {
                    codex_protocol::models::ReasoningItemReasoningSummary::SummaryText { text } => {
                        Some(text.clone())
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!joined.trim().is_empty()).then_some(joined)
        }
        ResponseItem::LocalShellCall { action, status, .. } => {
            Some(format!("local shell {status:?}: {action:?}"))
        }
        ResponseItem::FunctionCall {
            name, arguments, ..
        } => Some(format!("function call {name}({arguments})")),
        ResponseItem::FunctionCallOutput { output, .. } => Some(output.to_string()),
        ResponseItem::CustomToolCall {
            call_id,
            name,
            input,
            ..
        } => Some(format!("custom tool {call_id} {name}: {input}")),
        ResponseItem::CustomToolCallOutput { output, .. } => Some(output.to_string()),
        ResponseItem::ToolSearchCall {
            execution,
            arguments,
            ..
        } => Some(format!("tool search {execution}: {arguments}")),
        ResponseItem::ToolSearchOutput {
            status, execution, ..
        } => Some(format!("{status}: {execution}")),
        ResponseItem::WebSearchCall { action, .. } => Some(format!("web search: {action:?}")),
        ResponseItem::ImageGenerationCall { .. } => Some("image generation".to_string()),
        ResponseItem::GhostSnapshot { .. } => Some("ghost snapshot".to_string()),
        ResponseItem::Compaction { encrypted_content } => Some(encrypted_content.clone()),
        ResponseItem::Other => None,
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

#[cfg(test)]
mod tests {
    use super::LcmConfig;
    use super::OpenBrainConfig;
    use super::OpenBrainDurableMemoryCandidate;
    use super::OpenBrainGraphStatus;
    use super::OpenBrainGraphView;
    use super::OpenBrainProjectScopeStrategy;
    use super::OpenBrainRuntime;
    use super::content_items_to_text;
    use super::latest_unpromoted_summary_candidate_from_graph;
    use super::response_item_shape;
    use super::response_item_text;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;
    use std::collections::BTreeMap;
    use std::path::Path;

    #[test]
    fn default_open_brain_config_is_disabled_and_mirrors_durable_thoughts() {
        let config = OpenBrainConfig::default();
        assert!(!config.enabled);
        assert!(config.mirror_durable_to_thoughts);
        assert_eq!(
            config.project_scope_strategy,
            OpenBrainProjectScopeStrategy::GitRootFirst
        );
    }

    #[test]
    fn default_lcm_config_matches_repo_contract() {
        let config = LcmConfig::default();
        assert!(config.enabled);
        assert_eq!(config.fresh_tail_count, 64);
        assert_eq!(config.context_threshold, 0.75);
        assert_eq!(config.leaf_chunk_tokens, 20_000);
        assert_eq!(config.leaf_target_tokens, 1_200);
        assert_eq!(config.condensed_target_tokens, 2_000);
        assert_eq!(config.condensed_min_fanout, 4);
        assert_eq!(config.incremental_max_depth, 1);
        assert_eq!(config.max_expand_tokens, 4_000);
        assert_eq!(config.new_session_retain_depth, 2);
    }

    #[test]
    fn content_items_to_text_joins_non_empty_segments() {
        let items = vec![
            ContentItem::InputText {
                text: "hello".to_string(),
            },
            ContentItem::OutputText {
                text: String::new(),
            },
            ContentItem::OutputText {
                text: "world".to_string(),
            },
        ];

        assert_eq!(
            content_items_to_text(&items),
            Some("hello\nworld".to_string())
        );
    }

    #[test]
    fn response_item_text_renders_function_output() {
        let item = ResponseItem::FunctionCallOutput {
            call_id: "call-1".to_string(),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
        };

        assert_eq!(response_item_text(&item), Some("ok".to_string()));
        assert_eq!(
            response_item_shape(&item),
            (
                "function_call_output".to_string(),
                "tool".to_string(),
                None,
                Some("call-1".to_string())
            )
        );
    }

    #[test]
    fn context_token_budget_uses_threshold_and_cap() {
        unsafe {
            std::env::set_var("SUPABASE_URL", "https://example.supabase.co");
            std::env::set_var("SUPABASE_SERVICE_ROLE_KEY", "test-service-role");
        }
        let runtime = OpenBrainRuntime::from_env(
            &OpenBrainConfig {
                enabled: true,
                ..OpenBrainConfig::default()
            },
            &LcmConfig::default(),
            "session-1",
            "thread-1",
            "rollout-1",
            Path::new("/tmp/worktree"),
        )
        .expect("runtime");

        assert_eq!(runtime.context_token_budget(Some(20_000)), 4_000);
        assert_eq!(runtime.context_token_budget(Some(1_000)), 256);
    }

    #[test]
    fn latest_unpromoted_summary_candidate_prefers_latest_unpromoted_summary() {
        let graph = OpenBrainGraphView {
            thread_id: "thread-1".to_string(),
            project_key: None,
            scope_key: None,
            nodes: vec![
                serde_json::json!({
                    "node_id": "summary-new",
                    "node_kind": "summary_d0",
                    "title": "LCM leaf summary",
                    "content": "new summary content",
                    "source_event_ids": ["event-2", "event-3"],
                    "src_tok": 42_000,
                    "desc_tok": 1_200
                }),
                serde_json::json!({
                    "node_id": "summary-old",
                    "node_kind": "summary_d0",
                    "title": "older summary",
                    "content": "old summary content",
                    "source_event_ids": ["event-1"],
                    "src_tok": 21_000,
                    "desc_tok": 1_000
                }),
            ],
            edges: vec![serde_json::json!({
                "relationship_type": "PROMOTES_TO",
                "to_node_id": "summary-old"
            })],
            fresh_tail: Vec::new(),
            context_packets: Vec::new(),
            status: Some(OpenBrainGraphStatus {
                fresh_tail_count: 0,
                packet_count: 1,
                latest_packet_id: Some("packet-1".to_string()),
                latest_packet_token_budget: Some(4_000),
                latest_packet_query: Some("Operator console redesign".to_string()),
                node_kind_counts: BTreeMap::new(),
                depth_counts: BTreeMap::new(),
            }),
        };

        let candidate = latest_unpromoted_summary_candidate_from_graph(&graph);

        assert_eq!(
            candidate,
            Some(OpenBrainDurableMemoryCandidate {
                summary_node_id: "summary-new".to_string(),
                title: "Operator console redesign".to_string(),
                content: "new summary content".to_string(),
                source_event_ids: vec!["event-2".to_string(), "event-3".to_string()],
                source_token_count: Some(42_000),
                summary_token_count: Some(1_200),
            })
        );
    }

    #[test]
    fn latest_unpromoted_summary_candidate_returns_none_when_all_summaries_promoted() {
        let graph = OpenBrainGraphView {
            thread_id: "thread-1".to_string(),
            project_key: None,
            scope_key: None,
            nodes: vec![serde_json::json!({
                "node_id": "summary-only",
                "node_kind": "summary_d0",
                "title": "LCM leaf summary",
                "content": "summary",
                "source_event_ids": ["event-1"],
                "src_tok": 22_000,
                "desc_tok": 1_100
            })],
            edges: vec![serde_json::json!({
                "relationship_type": "PROMOTES_TO",
                "to_node_id": "summary-only"
            })],
            fresh_tail: Vec::new(),
            context_packets: Vec::new(),
            status: None,
        };

        assert_eq!(latest_unpromoted_summary_candidate_from_graph(&graph), None);
    }
}
