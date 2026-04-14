#![allow(clippy::expect_used)]

use anyhow::Result;
use codex_core::config::Config;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::built_in_model_providers;
use codex_open_brain::OpenBrainProjectScopeStrategy;
use codex_protocol::protocol::ContextEngine;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use reqwest::header::AUTHORIZATION;
use reqwest::header::CONTENT_TYPE;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use serde_json::Value;
use serial_test::serial;
use std::process::Command;

const FIRST_REPLY: &str = "FIRST_REPLY";
const SECOND_REPLY: &str = "SECOND_REPLY";
const FINAL_REPLY: &str = "FINAL_REPLY";

fn open_brain_env() -> Option<(&'static str, &'static str, String, String)> {
    let url = std::env::var("OPEN_BRAIN_E2E_SUPABASE_URL")
        .ok()
        .or_else(|| std::env::var("OPEN_BRAIN_SUPABASE_URL").ok())
        .or_else(|| Some("http://127.0.0.1:54321".to_string()))?;
    let key = std::env::var("OPEN_BRAIN_E2E_SERVICE_ROLE_KEY")
        .ok()
        .or_else(|| std::env::var("OPEN_BRAIN_SERVICE_ROLE_KEY").ok())
        .or_else(discover_local_service_role_key)?;
    let url_env = if std::env::var("OPEN_BRAIN_E2E_SUPABASE_URL").is_ok() {
        "OPEN_BRAIN_E2E_SUPABASE_URL"
    } else {
        "OPEN_BRAIN_SUPABASE_URL"
    };
    let key_env = if std::env::var("OPEN_BRAIN_E2E_SERVICE_ROLE_KEY").is_ok() {
        "OPEN_BRAIN_E2E_SERVICE_ROLE_KEY"
    } else {
        "OPEN_BRAIN_SERVICE_ROLE_KEY"
    };
    Some((url_env, key_env, url, key))
}

fn discover_local_service_role_key() -> Option<String> {
    let output = Command::new("docker")
        .args([
            "exec",
            "supabase_kong_open-brain",
            "sh",
            "-lc",
            "cat /home/kong/kong.yml | sed -n \"s/.*apikey == '\\(sb_secret_[^']*\\)'.*/\\1/p\" | head -n 1",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let key = String::from_utf8(output.stdout).ok()?;
    let trimmed = key.trim();
    (!trimmed.is_empty()).then_some(trimmed.to_string())
}

fn non_openai_model_provider(server: &wiremock::MockServer) -> ModelProviderInfo {
    let mut provider = built_in_model_providers(None)["openai"].clone();
    provider.name = "OpenAI (open-brain-lcm-test)".into();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider
}

fn auth_headers(service_role_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "apikey",
        HeaderValue::from_str(service_role_key).expect("apikey header"),
    );
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {service_role_key}")).expect("bearer header"),
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers
}

async fn rest_rows(
    client: &reqwest::Client,
    base_url: &str,
    service_role_key: &str,
    table: &str,
    query: &[(&str, String)],
) -> Result<Vec<Value>> {
    let url = format!("{}/rest/v1/{}", base_url.trim_end_matches('/'), table);
    let response = client
        .get(url)
        .headers(auth_headers(service_role_key))
        .query(query)
        .send()
        .await?
        .error_for_status()?;
    Ok(response.json::<Vec<Value>>().await?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires local open-brain supabase credentials and a running local backend"]
#[serial]
async fn live_open_brain_lcm_auto_compaction_writes_graph_artifacts() -> Result<()> {
    let Some((supabase_url_env, service_role_key_env, supabase_url, service_role_key)) =
        open_brain_env()
    else {
        eprintln!("skipping: OPEN_BRAIN_* supabase credentials are not set");
        return Ok(());
    };
    unsafe {
        std::env::set_var(supabase_url_env, &supabase_url);
        std::env::set_var(service_role_key_env, &service_role_key);
    }

    let server = start_mock_server().await;
    let turn_one = sse(vec![
        ev_assistant_message("m1", FIRST_REPLY),
        ev_completed_with_tokens("r1", 70_000),
    ]);
    let turn_two = sse(vec![
        ev_assistant_message("m2", SECOND_REPLY),
        ev_completed_with_tokens("r2", 330_000),
    ]);
    let turn_three = sse(vec![
        ev_assistant_message("m3", FINAL_REPLY),
        ev_completed_with_tokens("r3", 120),
    ]);
    let request_log = mount_sse_sequence(&server, vec![turn_one, turn_two, turn_three]).await;

    let model_provider = non_openai_model_provider(&server);
    let mut builder = test_codex().with_config(move |config: &mut Config| {
        config.model_provider = model_provider;
        config.context_engine = ContextEngine::OpenBrainLcm;
        config.model_auto_compact_token_limit = Some(200_000);
        config.open_brain.enabled = true;
        config.open_brain.supabase_url_env = Some(supabase_url_env.to_string());
        config.open_brain.service_role_key_env = Some(service_role_key_env.to_string());
        config.open_brain.project_scope_strategy = OpenBrainProjectScopeStrategy::GitRootFirst;
        config.open_brain.mirror_durable_to_thoughts = true;
        config.lcm.enabled = true;
        config.lcm.fresh_tail_count = 2;
    });
    let test = builder.build(&server).await?;
    let session_id = test.session_configured.session_id.to_string();

    test.submit_turn("token limit start").await?;
    test.submit_turn("token limit push").await?;
    test.submit_turn("post auto follow-up").await?;

    let request_count = request_log.requests().len();
    assert_eq!(
        request_count, 3,
        "open_brain_lcm should compact locally without an extra model summarization request"
    );

    let client = reqwest::Client::new();
    let session_rows = rest_rows(
        &client,
        &supabase_url,
        &service_role_key,
        "ob_sessions",
        &[("session_id", format!("eq.{session_id}"))],
    )
    .await?;
    assert_eq!(session_rows.len(), 1, "expected one Open Brain session row");

    let event_rows = rest_rows(
        &client,
        &supabase_url,
        &service_role_key,
        "ob_events",
        &[
            ("session_id", format!("eq.{session_id}")),
            (
                "select",
                "event_id,event_sequence,item_kind,turn_id,thread_id".to_string(),
            ),
        ],
    )
    .await?;
    assert!(
        !event_rows.is_empty(),
        "expected synced Open Brain rollout items for the live session"
    );

    let node_rows = rest_rows(
        &client,
        &supabase_url,
        &service_role_key,
        "ob_nodes",
        &[
            ("session_id", format!("eq.{session_id}")),
            (
                "select",
                "node_id,node_type,node_kind,metadata,thread_id,project_key".to_string(),
            ),
            ("order", "created_at.desc".to_string()),
        ],
    )
    .await?;
    let summary_node = node_rows
        .iter()
        .find(|row| row["node_kind"] == "summary_d0")
        .expect("expected at least one leaf summary node");
    assert!(
        summary_node["metadata"]["span_id"].as_str().is_some(),
        "expected the leaf summary node to carry a compaction span id"
    );
    let context_packet = node_rows
        .iter()
        .find(|row| row["node_kind"] == "context_packet")
        .expect("expected a context packet node");
    assert!(
        summary_node["metadata"]["span_id"] != context_packet["metadata"]["packet_id"],
        "compaction span id should not alias the context packet id"
    );
    assert_eq!(
        context_packet["thread_id"].as_str(),
        Some(session_id.as_str()),
        "context packet should be scoped to the Codex thread id"
    );
    let packet_item_count = context_packet["metadata"]["item_count"]
        .as_u64()
        .unwrap_or_default();
    assert!(
        packet_item_count > 0,
        "context packet should contain at least one assembled packet item"
    );
    let span_id = summary_node["metadata"]["span_id"]
        .as_str()
        .expect("summary span id")
        .to_string();
    let span_rows = rest_rows(
        &client,
        &supabase_url,
        &service_role_key,
        "ob_spans",
        &[
            ("session_id", format!("eq.{session_id}")),
            ("span_id", format!("eq.{span_id}")),
            (
                "select",
                "span_id,start_event_sequence,end_event_sequence,removed_message_count,preserved_tail_count".to_string(),
            ),
        ],
    )
    .await?;
    assert_eq!(span_rows.len(), 1, "expected one matching compaction span");
    let span = &span_rows[0];
    assert!(
        span["start_event_sequence"].as_u64().is_some()
            && span["end_event_sequence"].as_u64().is_some(),
        "expected the compaction span to resolve a concrete event range"
    );
    assert!(
        span["removed_message_count"].as_u64().unwrap_or_default() > 0,
        "expected the compaction span to cover at least one compacted source item"
    );

    eprintln!("open_brain_lcm session_id={session_id}");
    Ok(())
}
