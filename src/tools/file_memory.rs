//! File-backed memory tools — plain text files under one directory that
//! survive across CLI runs.
//!
//! Ported from the AgentCpp `MemoryStore` + `MemoryRead/Write/List` tools.
//! Distinct from `crate::memory::MemoryManager`, which is a richer in-memory
//! conversation/history manager. This is the simpler "scratchpad" memory the
//! C++ build exposes to the model.
//!
//! Default root resolution (when constructed with `None`):
//!   $AGENTRUST_MEMORY_DIR  if set
//!   $XDG_DATA_HOME/agentrust/memory  on Linux/Unix
//!   $HOME/.agentrust/memory          fallback
//!   %APPDATA%\agentrust\memory       on Windows
//!
//! Path safety: names containing `..`, absolute prefixes, or any component
//! starting with `.` are rejected.

use super::{Tool, ToolError, ToolOutput};
use async_trait::async_trait;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

/// One entry in the file-backed memory store.
#[derive(Debug, Clone)]
pub struct FileMemoryEntry {
    /// Path relative to root, using forward slashes (e.g. "notes/projects.md").
    pub name: String,
    /// First non-empty line of the file (trimmed).
    pub summary: String,
    /// Raw size in bytes on disk.
    pub size_bytes: u64,
    /// Absolute path to the file.
    pub full_path: PathBuf,
}

/// Disk-backed memory store. Each entry is a plain text file under `root`.
#[derive(Clone)]
pub struct FileMemoryStore {
    root: PathBuf,
    ready: bool,
}

impl FileMemoryStore {
    /// Construct a new store. If `root` is `None`, the default location is
    /// used. If `create` is true the directory is created if missing.
    pub fn new(root: Option<PathBuf>, create: bool) -> Self {
        let root = root.unwrap_or_else(Self::default_root);
        let mut ready = false;
        if create && !root.exists() {
            let _ = std::fs::create_dir_all(&root);
        }
        if root.exists() && root.is_dir() {
            ready = true;
        }
        Self { root, ready }
    }

    /// Resolve the default root path described in the docs above.
    pub fn default_root() -> PathBuf {
        if let Ok(v) = std::env::var("AGENTRUST_MEMORY_DIR") {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
        #[cfg(windows)]
        {
            if let Ok(appdata) = std::env::var("APPDATA") {
                if !appdata.is_empty() {
                    return PathBuf::from(appdata).join("agentrust").join("memory");
                }
            }
            if let Ok(home) = std::env::var("USERPROFILE") {
                if !home.is_empty() {
                    return PathBuf::from(home).join(".agentrust").join("memory");
                }
            }
        }
        #[cfg(not(windows))]
        {
            if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
                if !xdg.is_empty() {
                    return PathBuf::from(xdg).join("agentrust").join("memory");
                }
            }
            if let Ok(home) = std::env::var("HOME") {
                if !home.is_empty() {
                    return PathBuf::from(home).join(".agentrust").join("memory");
                }
            }
        }
        PathBuf::from(".agentrust").join("memory")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Validate `name` and return the absolute path it resolves to.
    /// Returns `None` for any invalid name (traversal, absolute, hidden, …).
    pub fn resolve(&self, name: &str) -> Option<PathBuf> {
        if !self.ready {
            return None;
        }
        if !name_is_safe(name) {
            return None;
        }
        let candidate = self.root.join(name);
        // Defense in depth — the canonical path must still live under root.
        let canon = canonicalize_weakly(&candidate);
        let root_canon = canonicalize_weakly(&self.root);
        if !canon.starts_with(&root_canon) {
            return None;
        }
        Some(canon)
    }

    /// List every entry under the root, sorted by name.
    pub fn list(&self) -> Vec<FileMemoryEntry> {
        let mut out = Vec::new();
        if !self.ready {
            return out;
        }
        for entry in WalkDir::new(&self.root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !(name.starts_with('.') && e.depth() > 0)
            })
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let rel = match path.strip_prefix(&self.root) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(FileMemoryEntry {
                name: rel_str,
                summary: first_non_empty_line(path, 120),
                size_bytes: size,
                full_path: path.to_path_buf(),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Read the full contents of an entry. Returns `None` if not found.
    pub fn read(&self, name: &str) -> Option<String> {
        let path = self.resolve(name)?;
        if !path.is_file() {
            return None;
        }
        std::fs::read_to_string(path).ok()
    }

    /// Overwrite (or create) an entry. Returns the absolute path on success.
    pub fn write(&self, name: &str, content: &str) -> Option<PathBuf> {
        let path = self.resolve(name)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok()?;
        }
        std::fs::write(&path, content).ok()?;
        Some(path)
    }

    /// Remove an entry. Returns true on success.
    pub fn remove(&self, name: &str) -> bool {
        match self.resolve(name) {
            Some(p) => std::fs::remove_file(p).is_ok(),
            None => false,
        }
    }
}

impl Default for FileMemoryStore {
    fn default() -> Self {
        Self::new(None, true)
    }
}

fn canonicalize_weakly(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

fn name_is_safe(name: &str) -> bool {
    if name.is_empty() || name.len() > 512 {
        return false;
    }
    let first = name.chars().next().unwrap();
    if first == '/' || first == '\\' {
        return false;
    }
    // Windows drive prefix (e.g. "C:\…")
    let bytes = name.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        return false;
    }

    let mut current = String::new();
    let mut flush = |comp: &str| -> bool {
        if comp.is_empty() {
            return false;
        }
        if comp == "." || comp == ".." {
            return false;
        }
        if comp.starts_with('.') {
            return false;
        }
        for c in comp.chars() {
            if c == '\0' {
                return false;
            }
            if (c as u32) < 0x20 {
                return false;
            }
        }
        true
    };

    for c in name.chars() {
        if c == '/' || c == '\\' {
            if !flush(&current) {
                return false;
            }
            current.clear();
        } else {
            current.push(c);
        }
    }
    flush(&current)
}

fn first_non_empty_line(p: &Path, max_chars: usize) -> String {
    let f = match std::fs::File::open(p) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let reader = BufReader::new(f);
    for line in reader.lines().flatten() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.chars().count() > max_chars {
            let mut s: String = t.chars().take(max_chars).collect();
            s.push_str("...");
            return s;
        }
        return t.to_string();
    }
    String::new()
}

// ──────────────────────────── Tools ─────────────────────────────────────────

pub struct MemoryListTool {
    store: Arc<FileMemoryStore>,
}

impl MemoryListTool {
    pub fn new(store: Arc<FileMemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryListTool {
    fn name(&self) -> &str {
        "MemoryList"
    }
    fn description(&self) -> &str {
        "List every memory entry with its name, size and first-line summary. Memory entries are plain text files that survive across CLI runs."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        if !self.store.is_ready() {
            return Err(ToolError {
                message: format!(
                    "Memory store not ready at {}",
                    self.store.root().display()
                ),
                code: Some("memory_unready".to_string()),
            });
        }
        let entries = self.store.list();
        let mut out = format!(
            "Memory ({} entr{}, root={}):\n",
            entries.len(),
            if entries.len() == 1 { "y" } else { "ies" },
            self.store.root().display()
        );
        if entries.is_empty() {
            out.push_str("(empty)");
        } else {
            for e in &entries {
                out.push_str(&format!(
                    "  {}  ({}B)  {}\n",
                    e.name, e.size_bytes, e.summary
                ));
            }
        }
        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: out,
            metadata: HashMap::new(),
        })
    }
}

pub struct MemoryReadTool {
    store: Arc<FileMemoryStore>,
}

impl MemoryReadTool {
    pub fn new(store: Arc<FileMemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn name(&self) -> &str {
        "MemoryRead"
    }
    fn description(&self) -> &str {
        "Read one memory entry by its relative name (e.g. 'notes/projects.md')."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Relative path under the memory root." }
            },
            "required": ["name"]
        })
    }
    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let name = input["name"].as_str().unwrap_or("");
        if name.is_empty() {
            return Err(ToolError {
                message: "'name' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        match self.store.read(name) {
            Some(body) => Ok(ToolOutput {
                output_type: "text".to_string(),
                content: body,
                metadata: HashMap::new(),
            }),
            None => Err(ToolError {
                message: format!("memory entry not found or invalid name: {}", name),
                code: Some("not_found".to_string()),
            }),
        }
    }
}

pub struct MemoryWriteTool {
    store: Arc<FileMemoryStore>,
    read_only: bool,
}

impl MemoryWriteTool {
    pub fn new(store: Arc<FileMemoryStore>) -> Self {
        Self { store, read_only: false }
    }
    pub fn with_read_only(store: Arc<FileMemoryStore>, read_only: bool) -> Self {
        Self { store, read_only }
    }
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn name(&self) -> &str {
        "MemoryWrite"
    }
    fn description(&self) -> &str {
        "Create or overwrite a memory entry. Names are relative paths under the memory root."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name":    { "type": "string", "description": "Relative path under the memory root." },
                "content": { "type": "string", "description": "Full text body to store." }
            },
            "required": ["name", "content"]
        })
    }
    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        if self.read_only {
            return Err(ToolError {
                message: "read-only mode: MemoryWrite is disabled".to_string(),
                code: Some("read_only".to_string()),
            });
        }
        let name = input["name"].as_str().unwrap_or("");
        let content = input["content"].as_str().unwrap_or("");
        if name.is_empty() {
            return Err(ToolError {
                message: "'name' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        match self.store.write(name, content) {
            Some(p) => Ok(ToolOutput {
                output_type: "text".to_string(),
                content: format!("Wrote {} bytes to {}", content.len(), p.display()),
                metadata: HashMap::new(),
            }),
            None => Err(ToolError {
                message: format!("memory write failed (invalid name or IO error): {}", name),
                code: Some("write_failed".to_string()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_safety_basics() {
        assert!(name_is_safe("a.md"));
        assert!(name_is_safe("notes/projects.md"));
        assert!(!name_is_safe(""));
        assert!(!name_is_safe("/etc/passwd"));
        assert!(!name_is_safe("../escape"));
        assert!(!name_is_safe(".hidden"));
        assert!(!name_is_safe("notes/.hidden"));
        assert!(!name_is_safe("notes//double"));
        assert!(!name_is_safe("C:\\windows"));
    }
}
