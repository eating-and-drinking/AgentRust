//! Computer Use tool — coarse screen control via shell-out.
//!
//! Ported from the AgentCpp `ComputerTool`. Exposes the following actions:
//!
//!   screenshot, cursor_position, mouse_move, left_click, right_click,
//!   middle_click, double_click, scroll, type, key
//!
//! Backend per platform:
//!   - Linux:  `xdotool` for input, `scrot`/`maim` for screenshots
//!   - macOS:  `cliclick` for input, `screencapture` for screenshots
//!   - Windows / other: not supported
//!
//! Screenshots are base64-encoded into the tool output metadata, so a
//! computer-use-aware loop can surface them back to the model as image
//! content blocks. The textual content field carries a human-readable note
//! (path + size).

use super::{Tool, ToolError, ToolOutput};
use async_trait::async_trait;
use base64::Engine;
use std::collections::HashMap;
use std::process::Command;

#[derive(Clone, Copy)]
enum Platform {
    Linux,
    MacOS,
    Other,
}

fn detect_platform() -> Platform {
    if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(target_os = "macos") {
        Platform::MacOS
    } else {
        Platform::Other
    }
}

fn is_mutating(action: &str) -> bool {
    action != "screenshot" && action != "cursor_position"
}

/// Run a shell command and capture combined stdout/stderr.
fn run_cmd(cmd: &str) -> (i32, String) {
    match Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(out) => {
            let code = out.status.code().unwrap_or(-1);
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&out.stdout));
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            (code, text)
        }
        Err(e) => (-1, format!("failed to spawn shell: {}", e)),
    }
}

fn shq(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

fn temp_png_path() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if cfg!(target_os = "windows") {
        format!("{}\\agentrust-screen-{}.png", std::env::temp_dir().display(), ms)
    } else {
        format!("/tmp/agentrust-screen-{}.png", ms)
    }
}

fn read_file_bin(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn ok_text(s: impl Into<String>) -> ToolOutput {
    ToolOutput {
        output_type: "text".to_string(),
        content: s.into(),
        metadata: HashMap::new(),
    }
}

fn err(msg: impl Into<String>, code: &str) -> ToolError {
    ToolError {
        message: msg.into(),
        code: Some(code.to_string()),
    }
}

fn capture_screenshot_linux() -> Result<ToolOutput, ToolError> {
    let path = temp_png_path();
    let (rc, out) = run_cmd(&format!("scrot {}", shq(&path)));
    let path = if rc != 0 {
        // Fallback to maim
        let path2 = temp_png_path();
        let (rc2, out2) = run_cmd(&format!("maim {}", shq(&path2)));
        if rc2 != 0 {
            return Err(err(
                format!(
                    "screenshot failed (need `scrot` or `maim` installed). {}{}",
                    out, out2
                ),
                "screenshot_failed",
            ));
        }
        path2
    } else {
        path
    };
    package_screenshot(&path)
}

fn capture_screenshot_macos() -> Result<ToolOutput, ToolError> {
    let path = temp_png_path();
    let (rc, out) = run_cmd(&format!("screencapture -x {}", shq(&path)));
    if rc != 0 {
        return Err(err(
            format!("screencapture failed: {}", out),
            "screenshot_failed",
        ));
    }
    package_screenshot(&path)
}

fn package_screenshot(path: &str) -> Result<ToolOutput, ToolError> {
    let bin = read_file_bin(path).ok_or_else(|| {
        err(
            format!("screenshot file empty/unreadable: {}", path),
            "screenshot_failed",
        )
    })?;
    let size = bin.len();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bin);
    let mut metadata = HashMap::new();
    metadata.insert("image_base64".to_string(), serde_json::Value::String(b64));
    metadata.insert(
        "image_mime".to_string(),
        serde_json::Value::String("image/png".to_string()),
    );
    metadata.insert(
        "image_path".to_string(),
        serde_json::Value::String(path.to_string()),
    );
    Ok(ToolOutput {
        output_type: "image".to_string(),
        content: format!("screenshot {} ({} bytes)", path, size),
        metadata,
    })
}

fn exec_linux(action: &str, input: &serde_json::Value) -> Result<ToolOutput, ToolError> {
    if action == "screenshot" {
        return capture_screenshot_linux();
    }
    if action == "cursor_position" {
        let (rc, out) = run_cmd("xdotool getmouselocation --shell");
        if rc != 0 {
            return Err(err(format!("xdotool failed: {}", out), "xdotool_failed"));
        }
        return Ok(ok_text(out));
    }

    let mut xcmd = String::from("xdotool ");
    match action {
        "mouse_move" => {
            let x = input["x"].as_i64().unwrap_or(0);
            let y = input["y"].as_i64().unwrap_or(0);
            xcmd.push_str(&format!("mousemove {} {}", x, y));
        }
        "left_click" => xcmd.push_str("click 1"),
        "middle_click" => xcmd.push_str("click 2"),
        "right_click" => xcmd.push_str("click 3"),
        "double_click" => xcmd.push_str("click --repeat 2 --delay 100 1"),
        "scroll" => {
            let delta = input["delta"].as_i64().unwrap_or(1);
            let btn = if delta > 0 { 5 } else { 4 };
            let abs_d = delta.unsigned_abs();
            xcmd.push_str(&format!("click --repeat {} {}", abs_d, btn));
        }
        "type" => {
            let text = input["text"].as_str().unwrap_or("");
            xcmd.push_str(&format!("type --delay 12 -- {}", shq(text)));
        }
        "key" => {
            let keys = input["keys"].as_str().unwrap_or("");
            if keys.is_empty() {
                return Err(err("'keys' required for action=key", "missing_parameter"));
            }
            xcmd.push_str(&format!("key {}", shq(keys)));
        }
        _ => return Err(err(format!("unknown action: {}", action), "unknown_action")),
    }

    let (rc, out) = run_cmd(&xcmd);
    if rc != 0 {
        return Err(err(format!("xdotool failed: {}", out), "xdotool_failed"));
    }
    Ok(ok_text(format!("ok: {}", action)))
}

fn exec_macos(action: &str, input: &serde_json::Value) -> Result<ToolOutput, ToolError> {
    if action == "screenshot" {
        return capture_screenshot_macos();
    }
    if action == "cursor_position" {
        let (rc, out) = run_cmd("cliclick p");
        if rc != 0 {
            return Err(err(
                format!(
                    "cliclick failed (install via `brew install cliclick`): {}",
                    out
                ),
                "cliclick_failed",
            ));
        }
        return Ok(ok_text(out));
    }

    let mut ccmd = String::from("cliclick ");
    match action {
        "mouse_move" => {
            let x = input["x"].as_i64().unwrap_or(0);
            let y = input["y"].as_i64().unwrap_or(0);
            ccmd.push_str(&format!("m:{},{}", x, y));
        }
        "left_click" => {
            if input.get("x").is_some() && input.get("y").is_some() {
                let x = input["x"].as_i64().unwrap_or(0);
                let y = input["y"].as_i64().unwrap_or(0);
                ccmd.push_str(&format!("c:{},{}", x, y));
            } else {
                ccmd.push_str("c:.");
            }
        }
        "right_click" => ccmd.push_str("rc:."),
        "double_click" => ccmd.push_str("dc:."),
        "type" => {
            let text = input["text"].as_str().unwrap_or("");
            ccmd.push_str(&format!("t:{}", shq(text)));
        }
        "key" => {
            let keys = input["keys"].as_str().unwrap_or("");
            ccmd.push_str(&format!("kp:{}", shq(keys)));
        }
        "scroll" => {
            return Err(err(
                "scroll not supported on macOS in this version",
                "unsupported",
            ));
        }
        _ => return Err(err(format!("unknown action: {}", action), "unknown_action")),
    }

    let (rc, out) = run_cmd(&ccmd);
    if rc != 0 {
        return Err(err(
            format!(
                "cliclick failed (install via `brew install cliclick`): {}",
                out
            ),
            "cliclick_failed",
        ));
    }
    Ok(ok_text(format!("ok: {}", action)))
}

pub struct ComputerTool {
    read_only: bool,
}

impl Default for ComputerTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputerTool {
    pub fn new() -> Self {
        Self { read_only: false }
    }

    pub fn with_read_only(read_only: bool) -> Self {
        Self { read_only }
    }
}

#[async_trait]
impl Tool for ComputerTool {
    fn name(&self) -> &str {
        "Computer"
    }

    fn description(&self) -> &str {
        "Control the screen and input devices. Actions: screenshot, cursor_position, mouse_move, left_click, right_click, middle_click, double_click, scroll, type, key. Requires xdotool+scrot/maim on Linux, cliclick+screencapture on macOS. Windows is not supported. Screenshots are returned base64-encoded in tool metadata."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "screenshot | cursor_position | mouse_move | left_click | right_click | middle_click | double_click | scroll | type | key"
                },
                "x":     { "type": "integer", "description": "x coord (for mouse_move/click)" },
                "y":     { "type": "integer", "description": "y coord" },
                "text":  { "type": "string",  "description": "text to type" },
                "keys":  { "type": "string",  "description": "key combo, e.g. \"ctrl+c\", \"Return\"" },
                "delta": { "type": "integer", "description": "scroll amount (negative = up)" }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let action = input["action"].as_str().unwrap_or("");
        if action.is_empty() {
            return Err(err("'action' is required", "missing_parameter"));
        }

        if self.read_only && is_mutating(action) {
            return Err(err(
                format!("read-only mode: Computer action '{}' is disabled", action),
                "read_only",
            ));
        }

        match detect_platform() {
            Platform::Linux => exec_linux(action, &input),
            Platform::MacOS => exec_macos(action, &input),
            Platform::Other => Err(err(
                "Computer tool not supported on this platform (Linux/macOS only)",
                "unsupported_platform",
            )),
        }
    }
}
