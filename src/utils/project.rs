//! Project utilities

use std::path::PathBuf;

/// Initialize a new project
pub fn init_project(name: &str) -> anyhow::Result<()> {
    let project_dir = PathBuf::from(name);
    
    // Create project directory
    std::fs::create_dir_all(&project_dir)?;
    
    // Create basic structure
    std::fs::create_dir_all(project_dir.join("src"))?;
    
    // Create AGENT.md file — project context for any agent run inside this directory.
    let agent_md = format!(
        "# {}\n\nThis project was initialized with AgentRust.\n\n## Structure\n\n- `src/` — source code\n- `AGENT.md` — project context for the agent\n\n## Notes\n\nDescribe the project's goals, constraints, and conventions here so the\nagent can pick them up automatically on every run.\n",
        name
    );
    std::fs::write(project_dir.join("AGENT.md"), agent_md)?;
    
    // Create .gitignore
    let gitignore = "target/\n*.log\n.env\n";
    std::fs::write(project_dir.join(".gitignore"), gitignore)?;
    
    println!("Created project structure:");
    println!("  {}/", name);
    println!("    src/");
    println!("    AGENT.md");
    println!("    .gitignore");
    
    Ok(())
}

/// Detect project type
pub fn detect_project_type(path: &std::path::Path) -> ProjectType {
    // Check for various project markers
    if path.join("Cargo.toml").exists() {
        ProjectType::Rust
    } else if path.join("package.json").exists() {
        ProjectType::JavaScript
    } else if path.join("go.mod").exists() {
        ProjectType::Go
    } else if path.join("pyproject.toml").exists() || path.join("setup.py").exists() {
        ProjectType::Python
    } else if path.join("CMakeLists.txt").exists() {
        ProjectType::Cpp
    } else {
        ProjectType::Unknown
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectType {
    Rust,
    JavaScript,
    Python,
    Go,
    Cpp,
    Unknown,
}

impl std::fmt::Display for ProjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectType::Rust => write!(f, "Rust"),
            ProjectType::JavaScript => write!(f, "JavaScript/TypeScript"),
            ProjectType::Python => write!(f, "Python"),
            ProjectType::Go => write!(f, "Go"),
            ProjectType::Cpp => write!(f, "C/C++"),
            ProjectType::Unknown => write!(f, "Unknown"),
        }
    }
}