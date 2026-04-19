use anyhow::Context;
use codex_app_server_protocol::UserInput;
use codex_config::types::ExternalContextConfig;
use codex_core::ThreadConfigSnapshot;
use codex_login::default_client::build_reqwest_client;
use serde::Deserialize;
use serde::Serialize;
use std::time::Duration;

#[derive(Serialize)]
struct ExternalContextRequest<'a> {
    thread_id: &'a str,
    model: &'a str,
    cwd: String,
    input: &'a [UserInput],
}

#[derive(Deserialize, Default)]
struct ExternalContextResponse {
    #[serde(default)]
    additional_contexts: Vec<String>,
}

pub(crate) async fn fetch_turn_start_context(
    config: &ExternalContextConfig,
    thread_id: &str,
    config_snapshot: &ThreadConfigSnapshot,
    input: &[UserInput],
) -> anyhow::Result<Vec<String>> {
    let Some(url) = config.url.as_deref() else {
        return Ok(Vec::new());
    };

    let mut request = build_reqwest_client()
        .post(url)
        .timeout(Duration::from_millis(config.timeout_ms))
        .json(&ExternalContextRequest {
            thread_id,
            model: &config_snapshot.model,
            cwd: config_snapshot.cwd.display().to_string(),
            input,
        });

    if let Some(env_var) = config.bearer_token_env_var.as_deref() {
        let token = std::env::var(env_var)
            .with_context(|| format!("missing bearer token env var `{env_var}`"))?;
        let trimmed = token.trim();
        if trimmed.is_empty() {
            anyhow::bail!("bearer token env var `{env_var}` is empty");
        }
        request = request.bearer_auth(trimmed);
    }

    let response = request
        .send()
        .await
        .context("external context request failed")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let detail = body.trim();
        if detail.is_empty() {
            anyhow::bail!("external context provider returned HTTP {status}");
        }
        anyhow::bail!("external context provider returned HTTP {status}: {detail}");
    }

    let response: ExternalContextResponse = response
        .json()
        .await
        .context("external context response deserialization failed")?;

    Ok(response
        .additional_contexts
        .into_iter()
        .filter_map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect())
}
