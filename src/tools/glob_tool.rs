//! Glob tool — recursive pattern-based file search.
//!
//! Ported from the AgentCpp `GlobTool`. Supports:
//!   *   any chars except path separator
//!   ?   single char
//!   **  any number of path components
//!   []  character classes (with optional `!` negation)
//!
//! Hidden directories (starting with `.`) and a few high-noise directories
//! (`node_modules`, `.git`, `build`, `dist`, `target`) are skipped.
//!
//! Results are sorted by path and capped at `limit` (default 100, max 1000).

use super::{Tool, ToolError, ToolOutput};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct GlobTool;

impl Default for GlobTool {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Searches recursively under the given path. Pattern supports * (any chars except /), ** (any path), ? (single char), and [] character classes. Returns up to 100 matching paths by default."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string",  "description": "Glob pattern to match (e.g. '**/*.rs', 'src/*.toml')." },
                "path":    { "type": "string",  "description": "Directory to search in. Defaults to current working directory." },
                "limit":   { "type": "integer", "description": "Maximum number of results (default 100).", "default": 100 }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let pattern = input["pattern"].as_str().unwrap_or("");
        if pattern.is_empty() {
            return Err(ToolError {
                message: "pattern is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        let limit: usize = input["limit"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(100)
            .min(1000);

        let base_str = input["path"].as_str().unwrap_or("");
        let base: PathBuf = if base_str.is_empty() {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else {
            let p = PathBuf::from(base_str);
            if p.is_relative() {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(p)
            } else {
                p
            }
        };

        if !base.exists() {
            return Err(ToolError {
                message: format!("Path does not exist: {}", base.display()),
                code: Some("path_not_found".to_string()),
            });
        }

        let mut matches = glob_walk(&base, pattern, limit);
        matches.sort();

        if matches.is_empty() {
            return Ok(ToolOutput {
                output_type: "text".to_string(),
                content: format!("No files found matching: {}", pattern),
                metadata: HashMap::new(),
            });
        }

        let mut out = format!("{} file(s) matching \"{}\":\n\n", matches.len(), pattern);
        for p in &matches {
            let rel = p.strip_prefix(&base).unwrap_or(p);
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            out.push_str(&rel_str);
            out.push('\n');
        }
        if matches.len() >= limit {
            out.push_str(&format!(
                "\n(Results capped at {}. Use a more specific pattern.)",
                limit
            ));
        }

        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: out,
            metadata: HashMap::new(),
        })
    }
}

// ───────────────────────── glob matcher ──────────────────────────────────────

/// Match a single path component against a pattern component.
/// Supports `*`, `?`, `[abc]`, `[!abc]`.
fn match_component(pattern: &str, s: &str) -> bool {
    let pat = pattern.as_bytes();
    let st = s.as_bytes();
    matcher(pat, 0, st, 0)
}

fn matcher(pat: &[u8], mut pi: usize, st: &[u8], mut si: usize) -> bool {
    while pi < pat.len() && si < st.len() {
        let p = pat[pi];
        if p == b'*' {
            pi += 1;
            if pi == pat.len() {
                // remaining input must contain no separator
                return !st[si..].contains(&b'/');
            }
            while si <= st.len() {
                if si > 0 && st[si - 1] == b'/' {
                    // can't cross a slash
                    return false;
                }
                if matcher(pat, pi, st, si) {
                    return true;
                }
                if si == st.len() {
                    return false;
                }
                if st[si] == b'/' {
                    return false;
                }
                si += 1;
            }
            return false;
        } else if p == b'?' {
            if st[si] == b'/' {
                return false;
            }
            pi += 1;
            si += 1;
        } else if p == b'[' {
            if let Some(close) = pat[pi + 1..].iter().position(|&b| b == b']') {
                let close_abs = pi + 1 + close;
                let mut start = pi + 1;
                let negate = pat[start] == b'!';
                if negate {
                    start += 1;
                }
                let cls = &pat[start..close_abs];
                let found = cls.contains(&st[si]);
                if negate == found {
                    return false;
                }
                pi = close_abs + 1;
                si += 1;
            } else {
                if p != st[si] {
                    return false;
                }
                pi += 1;
                si += 1;
            }
        } else if p == st[si] {
            pi += 1;
            si += 1;
        } else {
            return false;
        }
    }
    // Consume trailing '*'s
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len() && si == st.len()
}

/// Match a full relative path against a glob pattern supporting `**`.
fn glob_match(pattern: &str, rel_path: &str) -> bool {
    let ppat: Vec<&str> = pattern.split('/').collect();
    let ppath: Vec<&str> = rel_path.split('/').collect();

    fn rec(ppat: &[&str], pi: usize, ppath: &[&str], si: usize) -> bool {
        if pi == ppat.len() {
            return si == ppath.len();
        }
        if ppat[pi] == "**" {
            if rec(ppat, pi + 1, ppath, si) {
                return true;
            }
            for k in si..ppath.len() {
                if rec(ppat, pi + 1, ppath, k + 1) {
                    return true;
                }
            }
            return false;
        }
        if si == ppath.len() {
            return false;
        }
        if !match_component(ppat[pi], ppath[si]) {
            return false;
        }
        rec(ppat, pi + 1, ppath, si + 1)
    }

    rec(&ppat, 0, &ppath, 0)
}

fn glob_walk(base: &Path, pattern: &str, limit: usize) -> Vec<PathBuf> {
    let mut results = Vec::new();

    let walker = WalkDir::new(base)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            if name.starts_with('.') && e.depth() > 0 {
                return false;
            }
            !matches!(
                name.as_ref(),
                "node_modules" | "build" | "dist" | "target"
            )
        });

    for entry in walker.filter_map(|e| e.ok()) {
        if results.len() >= limit {
            break;
        }
        let p = entry.path();
        if p == base {
            continue;
        }
        let rel = match p.strip_prefix(base) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if glob_match(pattern, &rel_str) {
            results.push(p.to_path_buf());
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_matches_single_component() {
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "a/b.rs"));
    }

    #[test]
    fn double_star_crosses_components() {
        assert!(glob_match("**/*.rs", "a/b/c.rs"));
        assert!(glob_match("**/*.rs", "lib.rs"));
    }

    #[test]
    fn question_matches_one_char() {
        assert!(glob_match("?.rs", "a.rs"));
        assert!(!glob_match("?.rs", "ab.rs"));
    }

    #[test]
    fn char_class() {
        assert!(glob_match("[ab].rs", "a.rs"));
        assert!(glob_match("[ab].rs", "b.rs"));
        assert!(!glob_match("[ab].rs", "c.rs"));
        assert!(glob_match("[!ab].rs", "c.rs"));
    }
}
