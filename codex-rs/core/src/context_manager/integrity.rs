use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CallOutputFamily {
    FunctionOrShell,
    CustomTool,
    ClientToolSearch,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PairKey {
    family: CallOutputFamily,
    call_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HistoryIntegrityIssue {
    MissingOutput {
        call_id: String,
        family: &'static str,
    },
    OrphanOutput {
        call_id: String,
        family: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HistoryRepairResult {
    pub(crate) items: Vec<ResponseItem>,
    pub(crate) issues: Vec<HistoryIntegrityIssue>,
    pub(crate) changed: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct MissingOutputInsertion {
    index: usize,
    item: ResponseItem,
}

#[derive(Clone, Debug, Default)]
struct PairPresence {
    first_call_index: Option<usize>,
    first_output_index: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct HistoryIntegrityAnalysis {
    missing_outputs: Vec<MissingOutputInsertion>,
    orphan_output_indexes: Vec<usize>,
    issues: Vec<HistoryIntegrityIssue>,
    pair_presence: HashMap<PairKey, PairPresence>,
}

pub(crate) fn compute_protected_tail_split(
    items: &[ResponseItem],
    requested_fresh_tail_count: usize,
) -> usize {
    if items.is_empty() {
        return 0;
    }
    let analysis = analyze_history(items);
    let mut split_at = items
        .len()
        .saturating_sub(requested_fresh_tail_count.min(items.len()));

    loop {
        let mut expanded = split_at;
        for pair in analysis.pair_presence.values() {
            let tail_contains_call = pair.first_call_index.is_some_and(|idx| idx >= split_at);
            let tail_contains_output = pair.first_output_index.is_some_and(|idx| idx >= split_at);
            if !(tail_contains_call || tail_contains_output) {
                continue;
            }
            if let Some(call_index) = pair.first_call_index {
                expanded = expanded.min(call_index);
            }
            if let Some(output_index) = pair.first_output_index {
                expanded = expanded.min(output_index);
            }
        }
        if expanded == split_at {
            return split_at;
        }
        split_at = expanded;
    }
}

pub(crate) fn validate_replacement_history(
    items: &[ResponseItem],
) -> Result<(), Vec<HistoryIntegrityIssue>> {
    let issues = analyze_history(items).issues;
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub(crate) fn repair_replacement_history_for_resume(items: &[ResponseItem]) -> HistoryRepairResult {
    let analysis = analyze_history(items);
    if analysis.issues.is_empty() {
        return HistoryRepairResult {
            items: items.to_vec(),
            issues: Vec::new(),
            changed: false,
        };
    }

    let mut repaired = items.to_vec();
    for orphan_index in analysis.orphan_output_indexes.iter().rev() {
        repaired.remove(*orphan_index);
    }
    for missing in analysis.missing_outputs.iter().rev() {
        repaired.insert(missing.index + 1, missing.item.clone());
    }

    HistoryRepairResult {
        items: repaired,
        issues: analysis.issues,
        changed: true,
    }
}

pub(crate) fn missing_output_insertions(items: &[ResponseItem]) -> Vec<(usize, ResponseItem)> {
    analyze_history(items)
        .missing_outputs
        .into_iter()
        .map(|missing| (missing.index, missing.item))
        .collect()
}

pub(crate) fn orphan_output_indexes(items: &[ResponseItem]) -> Vec<usize> {
    analyze_history(items).orphan_output_indexes
}

fn analyze_history(items: &[ResponseItem]) -> HistoryIntegrityAnalysis {
    let mut pair_presence = HashMap::<PairKey, PairPresence>::new();

    for (index, item) in items.iter().enumerate() {
        if let Some(key) = call_key(item) {
            pair_presence.entry(key).or_default().first_call_index = Some(index);
        }
        if let Some(key) = output_key(item) {
            pair_presence.entry(key).or_default().first_output_index = Some(index);
        }
    }

    let mut analysis = HistoryIntegrityAnalysis {
        pair_presence,
        ..Default::default()
    };

    for (index, item) in items.iter().enumerate() {
        if let Some(key) = call_key(item) {
            let presence = analysis
                .pair_presence
                .get(&key)
                .expect("pair presence should exist for call key");
            if presence.first_output_index.is_none() {
                analysis.issues.push(HistoryIntegrityIssue::MissingOutput {
                    call_id: key.call_id.clone(),
                    family: family_label(key.family),
                });
                analysis.missing_outputs.push(MissingOutputInsertion {
                    index,
                    item: synthetic_output_for_call(item),
                });
            }
        }

        if let Some(key) = output_key(item) {
            let presence = analysis
                .pair_presence
                .get(&key)
                .expect("pair presence should exist for output key");
            if presence.first_call_index.is_none() {
                analysis.issues.push(HistoryIntegrityIssue::OrphanOutput {
                    call_id: key.call_id.clone(),
                    family: family_label(key.family),
                });
                analysis.orphan_output_indexes.push(index);
            }
        }
    }

    analysis
}

fn call_key(item: &ResponseItem) -> Option<PairKey> {
    match item {
        ResponseItem::FunctionCall { call_id, .. } => Some(PairKey {
            family: CallOutputFamily::FunctionOrShell,
            call_id: call_id.clone(),
        }),
        ResponseItem::LocalShellCall {
            call_id: Some(call_id),
            ..
        } => Some(PairKey {
            family: CallOutputFamily::FunctionOrShell,
            call_id: call_id.clone(),
        }),
        ResponseItem::CustomToolCall { call_id, .. } => Some(PairKey {
            family: CallOutputFamily::CustomTool,
            call_id: call_id.clone(),
        }),
        ResponseItem::ToolSearchCall {
            call_id: Some(call_id),
            execution,
            ..
        } if execution == "client" => Some(PairKey {
            family: CallOutputFamily::ClientToolSearch,
            call_id: call_id.clone(),
        }),
        _ => None,
    }
}

fn output_key(item: &ResponseItem) -> Option<PairKey> {
    match item {
        ResponseItem::FunctionCallOutput { call_id, .. } => Some(PairKey {
            family: CallOutputFamily::FunctionOrShell,
            call_id: call_id.clone(),
        }),
        ResponseItem::CustomToolCallOutput { call_id, .. } => Some(PairKey {
            family: CallOutputFamily::CustomTool,
            call_id: call_id.clone(),
        }),
        ResponseItem::ToolSearchOutput {
            call_id: Some(call_id),
            execution,
            ..
        } if execution == "client" => Some(PairKey {
            family: CallOutputFamily::ClientToolSearch,
            call_id: call_id.clone(),
        }),
        _ => None,
    }
}

fn synthetic_output_for_call(item: &ResponseItem) -> ResponseItem {
    match item {
        ResponseItem::FunctionCall { call_id, .. }
        | ResponseItem::LocalShellCall {
            call_id: Some(call_id),
            ..
        } => ResponseItem::FunctionCallOutput {
            call_id: call_id.clone(),
            output: FunctionCallOutputPayload::from_text("aborted".to_string()),
        },
        ResponseItem::CustomToolCall { call_id, .. } => ResponseItem::CustomToolCallOutput {
            call_id: call_id.clone(),
            name: None,
            output: FunctionCallOutputPayload::from_text("aborted".to_string()),
        },
        ResponseItem::ToolSearchCall {
            call_id: Some(call_id),
            ..
        } => ResponseItem::ToolSearchOutput {
            call_id: Some(call_id.clone()),
            status: "completed".to_string(),
            execution: "client".to_string(),
            tools: Vec::new(),
        },
        _ => panic!("synthetic_output_for_call called with non-call item"),
    }
}

fn family_label(family: CallOutputFamily) -> &'static str {
    match family {
        CallOutputFamily::FunctionOrShell => "function_or_shell",
        CallOutputFamily::CustomTool => "custom_tool",
        CallOutputFamily::ClientToolSearch => "client_tool_search",
    }
}

#[cfg(test)]
mod tests {
    use super::HistoryIntegrityIssue;
    use super::compute_protected_tail_split;
    use super::repair_replacement_history_for_resume;
    use super::validate_replacement_history;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;

    fn function_call(call_id: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: "tool".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: call_id.to_string(),
        }
    }

    fn function_output(call_id: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            call_id: call_id.to_string(),
            output: FunctionCallOutputPayload::from_text("ok".to_string()),
        }
    }

    fn reasoning() -> ResponseItem {
        ResponseItem::Reasoning {
            id: String::new(),
            summary: Vec::new(),
            encrypted_content: None,
            content: None,
        }
    }

    #[test]
    fn protected_tail_split_keeps_separated_function_call_pairs() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: Vec::new(),
                end_turn: None,
                phase: None,
            },
            function_call("call-1"),
            reasoning(),
            function_output("call-1"),
        ];

        assert_eq!(compute_protected_tail_split(&items, 1), 1);
    }

    #[test]
    fn validate_replacement_history_rejects_orphan_outputs() {
        let issues = validate_replacement_history(&[function_output("call-1")])
            .expect_err("orphan output should be invalid");

        assert_eq!(
            issues,
            vec![HistoryIntegrityIssue::OrphanOutput {
                call_id: "call-1".to_string(),
                family: "function_or_shell",
            }]
        );
    }

    #[test]
    fn repair_replacement_history_for_resume_drops_orphans_and_inserts_missing_outputs() {
        let items = vec![function_call("call-1"), function_output("call-2")];

        let repaired = repair_replacement_history_for_resume(&items);

        assert!(repaired.changed);
        assert_eq!(
            repaired.issues,
            vec![
                HistoryIntegrityIssue::MissingOutput {
                    call_id: "call-1".to_string(),
                    family: "function_or_shell",
                },
                HistoryIntegrityIssue::OrphanOutput {
                    call_id: "call-2".to_string(),
                    family: "function_or_shell",
                },
            ]
        );
        assert_eq!(
            repaired.items,
            vec![
                function_call("call-1"),
                ResponseItem::FunctionCallOutput {
                    call_id: "call-1".to_string(),
                    output: FunctionCallOutputPayload::from_text("aborted".to_string()),
                },
            ]
        );
    }
}
