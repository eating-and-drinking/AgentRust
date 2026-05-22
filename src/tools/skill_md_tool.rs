//! Skill tool — looks up a SKILL.md by name and returns its body to the model.
//!
//! Ported from the AgentCpp `SkillTool` + `SkillRegistry`. This is the
//! markdown-skill (one folder + one `SKILL.md`) pattern: each skill directory
//! has a SKILL.md file with YAML frontmatter (`name`, `description`) plus a
//! markdown body. The agent advertises every loaded skill in its system
//! prompt; the model invokes a skill by calling this tool with the skill
//! name, which returns the full markdown body.
//!
//! Distinct from `crate::skills::SkillRegistry`, which manages programmatic
//! Rust skills (`/commit`, `/review`, ...).

use super::{Tool, ToolError, ToolOutput};
use async_trait::async_trait;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One markdown-based skill loaded from disk.
#[derive(Debug, Clone)]
pub struct MarkdownSkill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub skill_md_path: PathBuf,
    pub dir: PathBuf,
}

/// Registry of markdown skills loaded from one or more root directories.
#[derive(Default)]
pub struct MarkdownSkillRegistry {
    skills: Vec<MarkdownSkill>,
    by_name: HashMap<String, usize>,
}

impl MarkdownSkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every `<root>/<skill-name>/SKILL.md` under `root`.
    /// Returns the number of skills successfully loaded.
    pub fn add_root(&mut self, root: &Path) -> usize {
        if !root.exists() || !root.is_dir() {
            return 0;
        }
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut loaded = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let skill_md = path.join("SKILL.md");
            if !skill_md.is_file() {
                continue;
            }
            let parsed = match parse_skill_file(&skill_md) {
                Some(s) => s,
                None => continue,
            };
            // Collision check — first one wins.
            if self.by_name.contains_key(&parsed.name) {
                continue;
            }
            self.by_name.insert(parsed.name.clone(), self.skills.len());
            self.skills.push(parsed);
            loaded += 1;
        }
        loaded
    }

    pub fn find(&self, name: &str) -> Option<&MarkdownSkill> {
        self.by_name.get(name).map(|i| &self.skills[*i])
    }

    pub fn all(&self) -> &[MarkdownSkill] {
        &self.skills
    }
}

fn parse_skill_file(skill_md: &Path) -> Option<MarkdownSkill> {
    let raw = fs::read_to_string(skill_md).ok()?;
    let lines: Vec<&str> = raw.split('\n').map(|l| l.trim_end_matches('\r')).collect();

    // Find first non-empty line; must be `---`
    let mut i = 0usize;
    while i < lines.len() && lines[i].trim().is_empty() {
        i += 1;
    }
    if i >= lines.len() || lines[i].trim() != "---" {
        return None;
    }
    let fm_start = i + 1;

    // Find closing `---`
    let mut fm_end = fm_start;
    while fm_end < lines.len() && lines[fm_end].trim() != "---" {
        fm_end += 1;
    }
    if fm_end >= lines.len() {
        return None;
    }

    let fm_lines: Vec<&str> = lines[fm_start..fm_end].to_vec();
    let fields = parse_frontmatter(&fm_lines);

    let name = fields.get("name")?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let description = fields.get("description").cloned().unwrap_or_default();

    let mut body = String::new();
    for line in lines.iter().skip(fm_end + 1) {
        body.push_str(line);
        body.push('\n');
    }

    Some(MarkdownSkill {
        name,
        description,
        body,
        skill_md_path: skill_md.to_path_buf(),
        dir: skill_md.parent().unwrap_or(skill_md).to_path_buf(),
    })
}

/// Very small YAML frontmatter parser: `key: value`, plus block scalars
/// `key: |` / `key: >` whose continuation lines are indented.
fn parse_frontmatter(lines: &[&str]) -> HashMap<String, String> {
    let mut result = HashMap::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let colon = match line.find(':') {
            Some(c) => c,
            None => {
                i += 1;
                continue;
            }
        };
        let key = line[..colon].trim().to_string();
        let val = line[colon + 1..].trim().to_string();
        if key.is_empty() {
            i += 1;
            continue;
        }

        if val == "|" || val == ">" {
            let mut acc = String::new();
            i += 1;
            while i < lines.len() {
                let cont = lines[i];
                if !cont.is_empty()
                    && !cont.starts_with(' ')
                    && !cont.starts_with('\t')
                {
                    break;
                }
                if !acc.is_empty() {
                    acc.push('\n');
                }
                acc.push_str(cont.trim());
                i += 1;
            }
            result.insert(key, acc);
            continue;
        }

        // Strip surrounding quotes if present
        let stripped = if val.len() >= 2
            && ((val.starts_with('"') && val.ends_with('"'))
                || (val.starts_with('\'') && val.ends_with('\'')))
        {
            val[1..val.len() - 1].to_string()
        } else {
            val
        };
        result.insert(key, stripped);
        i += 1;
    }
    result
}

// ──────────────────────────── Tool ──────────────────────────────────────────

pub struct SkillMdTool {
    registry: Arc<MarkdownSkillRegistry>,
}

impl SkillMdTool {
    pub fn new(registry: Arc<MarkdownSkillRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for SkillMdTool {
    fn name(&self) -> &str {
        "Skill"
    }

    fn description(&self) -> &str {
        "Look up a markdown-based skill by name and return its full SKILL.md body. Each skill is one folder containing a SKILL.md with YAML frontmatter (name, description) plus a markdown body of instructions."
    }

    fn input_schema(&self) -> serde_json::Value {
        let names: Vec<String> = self.registry.all().iter().map(|s| s.name.clone()).collect();
        if names.is_empty() {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Skill name. (No skills are currently loaded.)" }
                },
                "required": ["name"]
            })
        } else {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name. Loaded skills: see system prompt.",
                        "enum": names
                    }
                },
                "required": ["name"]
            })
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let name = input["name"].as_str().unwrap_or("");
        if name.is_empty() {
            return Err(ToolError {
                message: "'name' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        match self.registry.find(name) {
            Some(s) => Ok(ToolOutput {
                output_type: "text".to_string(),
                content: s.body.clone(),
                metadata: HashMap::new(),
            }),
            None => Err(ToolError {
                message: format!("skill not found: {}", name),
                code: Some("not_found".to_string()),
            }),
        }
    }
}
