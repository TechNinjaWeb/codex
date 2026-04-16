//! Lightweight markdown normalization ahead of the TUI renderer.
//!
//! The current transcript renderer intentionally supports a focused markdown subset. GitHub-style
//! pipe tables are a common shape in model output, but when left untouched they degrade into raw
//! source because the downstream renderer does not build table structure. Normalize those tables
//! into ordinary markdown lists so links and inline emphasis inside cells still flow through the
//! existing renderer.

pub(crate) fn normalize_markdown(input: &str) -> String {
    normalize_pipe_tables(input)
}

fn normalize_pipe_tables(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut code_fence: Option<&str> = None;
    let mut index = 0usize;

    while index < lines.len() {
        let line = lines[index];
        if let Some(marker) = fence_marker(line) {
            if code_fence == Some(marker) {
                code_fence = None;
            } else if code_fence.is_none() {
                code_fence = Some(marker);
            }
            out.push(line.to_string());
            index += 1;
            continue;
        }

        if code_fence.is_none()
            && let Some((headers, rows, consumed)) = parse_table_block(&lines[index..])
        {
            out.extend(render_table_as_list(&headers, &rows));
            if lines
                .get(index + consumed)
                .is_some_and(|line| !line.trim().is_empty())
            {
                out.push(String::new());
            }
            index += consumed;
            continue;
        }

        out.push(line.to_string());
        index += 1;
    }

    let mut normalized = out.join("\n");
    if input.ends_with('\n') {
        normalized.push('\n');
    }
    normalized
}

fn fence_marker(line: &str) -> Option<&'static str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("```") {
        Some("```")
    } else if trimmed.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    }
}

fn parse_table_block(lines: &[&str]) -> Option<(Vec<String>, Vec<Vec<String>>, usize)> {
    if lines.len() < 3 {
        return None;
    }

    let headers = parse_pipe_row(lines[0])?;
    if headers.len() < 2 || !is_pipe_separator_row(lines[1], headers.len()) {
        return None;
    }

    let mut rows = Vec::new();
    let mut consumed = 2usize;
    for line in lines.iter().skip(2) {
        let Some(row) = parse_pipe_row(line) else {
            break;
        };
        if row.len() != headers.len() {
            break;
        }
        rows.push(row);
        consumed += 1;
    }

    (!rows.is_empty()).then_some((headers, rows, consumed))
}

fn parse_pipe_row(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.contains('|') {
        return None;
    }

    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut escaped = false;

    for ch in trimmed.chars() {
        if escaped {
            cell.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '|' => {
                cells.push(cell.trim().to_string());
                cell.clear();
            }
            _ => cell.push(ch),
        }
    }
    if escaped {
        cell.push('\\');
    }
    cells.push(cell.trim().to_string());

    if trimmed.starts_with('|') && cells.first().is_some_and(String::is_empty) {
        cells.remove(0);
    }
    if trimmed.ends_with('|') && cells.last().is_some_and(String::is_empty) {
        cells.pop();
    }

    (cells.len() >= 2).then_some(cells)
}

fn is_pipe_separator_row(line: &str, expected_columns: usize) -> bool {
    let Some(cells) = parse_pipe_row(line) else {
        return false;
    };
    if cells.len() != expected_columns {
        return false;
    }

    cells.iter().all(|cell| {
        let trimmed = cell.trim();
        !trimmed.is_empty()
            && trimmed.contains('-')
            && trimmed.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
    })
}

fn render_table_as_list(headers: &[String], rows: &[Vec<String>]) -> Vec<String> {
    let mut out = Vec::new();
    for (row_index, row) in rows.iter().enumerate() {
        for (cell_index, value) in row.iter().enumerate() {
            let label = header_label(headers.get(cell_index), cell_index);
            let prefix = if cell_index == 0 { "- " } else { "  " };
            out.push(format!("{prefix}{label}: {value}"));
        }
        if row_index + 1 < rows.len() {
            out.push(String::new());
        }
    }
    out
}

fn header_label(header: Option<&String>, index: usize) -> String {
    header
        .map(|header| header.trim())
        .filter(|header| !header.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Column {}", index + 1))
}

#[cfg(test)]
mod tests {
    use super::normalize_markdown;
    use pretty_assertions::assert_eq;

    #[test]
    fn normalizes_simple_pipe_table_into_nested_list() {
        let markdown = "\
| Project | Repo |\n\
|---|---|\n\
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) |\n\
| Letta | [letta-ai/letta](https://github.com/letta-ai/letta) |\n";

        let normalized = normalize_markdown(markdown);
        assert_eq!(
            normalized,
            "\
- Project: Mem0\n\
  Repo: [mem0ai/mem0](https://github.com/mem0ai/mem0)\n\
\n\
- Project: Letta\n\
  Repo: [letta-ai/letta](https://github.com/letta-ai/letta)\n"
        );
    }

    #[test]
    fn leaves_table_syntax_inside_code_fences_untouched() {
        let markdown = "\
```md\n\
| left | right |\n\
|---|---|\n\
| a | b |\n\
```\n";

        assert_eq!(normalize_markdown(markdown), markdown);
    }
}
