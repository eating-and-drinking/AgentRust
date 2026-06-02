# AgentRust

> 一个用 Rust 实现的**通用自主智能体运行时**。

AgentRust 目标是做一个通用的智能体，但实际上只是[eating-and-drinking](https://github.com/eating-and-drinking) 在吃吃喝喝之余做的一个 toy 项目，如果你发现有用，记得给我star，我还有很多核心代码没有上传，没有star，我就不上传了。

一个 Rust 工作区，三种入口：CLI 主程序 `agentrust`、egui 桌面端 `agentrust-gui`、axum Web 服务 `agentrust-web`，并可选编译为 WebAssembly。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust Edition](https://img.shields.io/badge/edition-2021-orange.svg)](https://www.rust-lang.org/)

---

## 通用智能体的核心抽象

| 抽象 | 在哪儿 | 干什么 |
| --- | --- | --- |
| `Goal` | `src/agent/goal.rs` | 自然语言目标 + 成功标准 + 上下文 + 截止时间 |
| `Persona` | `src/agent/persona.rs` | 内置 6 种角色 + `Custom(name)`；每种附带系统提示、建议的工具包、温度 |
| `AgentRunner` | `src/agent/runner.rs` | 无头的 `plan → act → reflect` 循环，最大步数可配 |
| `Bundle` | `src/tools/bundle.rs` | 工具能力包（coding / knowledge / desktop / web / communication） |
| `ChatMessage` 多模态 | `src/api/mod.rs` | `user_with_images(text, vec![ImageRef::from_file(p)?])` 自动序列化为 OpenAI parts 数组 |
| `MetacognitionEngine` | `src/metacognition/` | Bayesian 自信念 + EFE 决策 + CoT 监控，已挂入 REPL |
| `MemoryManager` | `src/memory/` | SQLite + BM25 + 向量混合召回 + LLM 整合（"dream"） |

### 内置 Persona

| Slug | 角色 | 默认工具包 |
| --- | --- | --- |
| `general` | 领域中立通用助手（默认） | knowledge, coding |
| `coder` | 资深软件工程师 | coding, knowledge |
| `researcher` | 研究助理，强调引用 | knowledge, web |
| `writer` | 写作伙伴 | knowledge |
| `analyst` | 数据分析师 | coding, knowledge |
| `operator` | 桌面/浏览器操作员 | desktop, web |
| `<任意 slug>` | 先查 `~/.agentrust/personas/<slug>.toml`，否则退回通用提示词 | knowledge |

### 自定义 Persona

把以下内容放到 `~/.agentrust/personas/marketing.toml`：

```toml
name = "Marketing Copywriter"
system_prompt = """
You write punchy B2B SaaS marketing copy. Match brand voice, lead with
the customer pain point, keep paragraphs under 3 sentences.
"""
bundles = ["knowledge", "web"]
temperature = 0.8
```

然后：

```bash
agentrust task --goal "重写官网首页 hero section" --persona marketing
```

`Persona::from_slug` 会把任何未知 slug 映射成 `Persona::Custom(slug)`，`profile()` 优先读盘上 TOML，失败才退回 stub。

### 工具能力 Bundle

| Bundle | 包含工具 |
| --- | --- |
| `coding` | `file_read` / `file_write` / `file_edit` / `list_files` / `search` / `Glob` / `execute_command` / `git_operations` |
| `knowledge` | `file_read` / `note_edit` / `task_management` / `MemoryList/Read/Write` / `ChannelPublish/Read/List` / `Skill` / `Task`（子代理） |
| `desktop` | `Computer`（截屏/点击/键入） |
| `web` | `http_fetch`（GET + 字节封顶） / `web_search`（DuckDuckGo HTML） / `file_read` / `file_write` |
| `communication` | `ChannelPublish/Read/List` / `Task` |

---

## 安装

```bash
git clone https://github.com/eating-and-drinking/AgentRust.git
cd AgentRust
cargo build --release
```

需要 Rust 1.75+（edition 2021）。

构建产物：
- `target/release/agentrust` — 主 CLI
- `target/release/agentrust-gui` — egui 桌面端（`gui-egui` 特性）
- `target/release/agentrust-web` — Web 服务（`web` 特性）

### Cargo 特性

| 特性 | 默认 | 启用内容 |
| --- | --- | --- |
| `gui-egui` | ✅ | `eframe`, `egui`, `egui_extras` |
| `i18n` | ✅ | `fluent`, `fluent-bundle`, `unic-langid`, `rust-embed` |
| `web` | ❌ | `axum`, `tower`, `tower-http`, `askama` |
| `wasm` | ❌ | `wasm-bindgen` 系列 |
| `full` | ❌ | 同时启用以上所有 |

```bash
# 纯 CLI（不含 GUI/i18n）
cargo build --release --no-default-features

# 启动 Web 服务
cargo run --release --features web --bin agentrust-web
```

---

## 配置

### API Key 与 Base URL

`src/config/api_config.rs` 按顺序读取：

```
1. ANTHROPIC_API_KEY
2. DASHSCOPE_API_KEY
3. DEEPSEEK_API_KEY
4. settings.api.api_key（配置文件）
```

`API_BASE_URL` 控制接口根（默认 `https://api.anthropic.com`）。

```env
ANTHROPIC_API_KEY=sk-ant-...
API_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-3-5-sonnet-20241022
RUST_LOG=agentrust=info
```

### 通用智能体相关配置（新）

`~/.agentrust/settings.json` 新增两项：

```json
{
  "default_persona": "general",
  "enabled_bundles": []
}
```

- `default_persona`：CLI 未传 `--persona` 时回落到此。
- `enabled_bundles`：`task` 命令的全局工具包白名单；为空时使用 persona 的建议。

也可用 CLI 设置：

```bash
agentrust config set default_persona researcher
agentrust config set enabled_bundles "knowledge,web"
```

### 模型别名

| 别名 | 实际模型 ID |
| --- | --- |
| `opus` | `claude-3-opus-20240229` |
| `sonnet` | `claude-3-5-sonnet-20241022`（默认） |
| `haiku` | `claude-3-5-haiku-20241022` |
| 其它 | 原样透传 |

---

## 快速开始

### 自主任务（核心入口）

```bash
# 通用 persona，knowledge + coding 工具
agentrust task --goal "总结 src/agent 模块对外暴露的 API，并列出潜在的破坏性改动"

# 研究员 persona，只开 knowledge + web 工具
agentrust task --goal "对比 BM25 与 dense retrieval 的优缺点，给出参考文献" \
               --persona researcher

# 桌面 operator，明确成功标准，打印 trace
agentrust task --goal "打开 VSCode 并新建一个 Rust 项目" \
               --persona operator \
               --criterion "看到 'Cargo.toml created' 字样" \
               --trace

# 自定义工具包（覆盖 persona 建议）
agentrust task --goal "..." --bundles coding,knowledge --max-steps 20
```

`task` 命令的输出包含：
- 目标 / persona / bundles / max_steps
- 可选的 `--trace`：每一步的 `assistant` / `tool→` / `←tool` 标签 + 截断后的 payload
- `stop_reason`：`AssistantFinal` / `MaxStepsReached` / `ApiError` / `ToolError` / `Cancelled`
- 最终回答

### 交互式 REPL

```bash
agentrust repl
agentrust repl --prompt "审查 src/api/ 的设计"
```

REPL 斜杠命令：`help`, `status`, `clear`, `history`, `reset`, `config`, `exit`（每个都有 `.` / `:` 简写）。

### 单次查询

```bash
agentrust query --prompt "用 Rust 写一个 LRU 缓存"
```

---

## 全部子命令

定义于 `src/cli/mod.rs`：

| 命令 | 子项 | 说明 |
| --- | --- | --- |
| **`task`** | `--goal --persona --bundles --max-steps --criterion --trace` | **自主任务循环（通用智能体核心入口）** |
| `repl` | `--prompt` | 交互式 REPL |
| `query` | `--prompt` | 单次查询 |
| `agent` | `--agent-type --prompt` | 调度内置子代理 |
| `config` | `show` / `set <k> <v>` / `reset` | 配置管理 |
| `mcp` | `list` / `add` / `remove` / `restart` | MCP 服务管理 |
| `plugin` | `list/install/remove/update/search/enable/disable` | 插件市场 |
| `memory` | `status/clear/export/import/dream/auto-dream` | 记忆操作 |
| `services` | `status/start/stop/...` | 后台服务 |
| `magic-docs` | `list/check/update/clear` | Magic Docs |
| `team-sync` | `status/auth/sync/list/create/delete` | 团队记忆同步 |
| `skills` | `list/execute/help/search` | 技能管理 |
| `voice` | `--push-to-talk` | 语音输入（未实现） |
| `init` | `--name` | 初始化新项目（生成 `AGENT.md`） |
| `stress-test` | `--concurrency --iterations` | 并发压测 |
| `update` | — | 占位 |

---

## 多模态消息

```rust
use agentrust::api::{ApiClient, ChatMessage, ImageRef};

let img = ImageRef::from_file("screenshot.png")?;
let msg = ChatMessage::user_with_images(
    "What's on the screen?",
    vec![img],
);

let resp = api.chat(vec![msg], None).await?;
```

`ChatMessage::with_image(...)` 也可在已有消息上 builder 式追加图像。当 `images` 非空时，`ChatMessage` 的自定义 `Serialize` 实现会把 `content` 输出为 OpenAI 视觉端点期望的 parts 数组：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's on the screen?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

`images` 为空时序列化为普通字符串内容，向后兼容所有现有代码。

---

## 记忆 / 元认知 / MCP

（保留原有的成熟模块，与通用化改造正交）

- **记忆**：`agentrust memory dream` 触发一次 LLM 驱动整合；`AutoDreamService` 按"距上次 N 小时 + 累计 N 个会话"门槛触发后台整合。
- **元认知**：`MetacognitionEngine` 同时挂在 REPL 和 `task` 命令上。EFE 五动作 `Act / Reflect / Decompose / Escalate / Abort`，`Abort` 会立即中止 `AgentRunner` 循环并把 `stop_reason` 标为 `Cancelled`；`injection` 字段作为一次性 system 消息插入下一轮请求。可通过 `agentrust config set metacog.enabled false` 关掉。
- **MCP**：`agentrust mcp add filesystem --path /workspace` 走内置快路径；外部 stdio MCP 服务通过 `agentrust mcp add <name> "<cmd>"` 启动。

---

## 项目布局

```
AgentRust/
├── Cargo.toml
├── .env.example
├── locales/                # en.ftl, zh.ftl
├── src/
│   ├── main.rs / lib.rs    # 二进制 / 库入口
│   ├── agent/              # ★ 通用 Agent 运行时（Goal / Persona / Runner）
│   ├── cli/                # CLI 解析 + REPL
│   ├── api/                # OpenAI 兼容客户端 + 多模态消息
│   ├── config/             # Settings / ApiConfig（新增 persona/bundles 字段）
│   ├── state/              # AppState
│   ├── mcp/                # MCP JSON-RPC server + transport
│   ├── tools/              # 18 个工具 + bundle.rs 能力包
│   ├── memory/             # SQLite 记忆引擎
│   ├── metacognition/      # 自信念 / EFE / CoT 监控
│   ├── plugins/            # 插件加载/隔离
│   ├── services/           # AutoDream / MagicDocs / TeamSync / ...
│   ├── session/            # 会话管理
│   ├── skills/             # 技能注册
│   ├── channels/           # 进程内 pub/sub
│   ├── terminal/           # 终端渲染
│   ├── voice/              # 语音（占位）
│   ├── advanced/           # SSH / 远程执行 / 脚手架
│   ├── utils/              # 公共工具
│   ├── gui/                # eframe + egui 桌面端
│   ├── web/                # axum 插件市场
│   ├── wasm/               # wasm-bindgen 绑定
│   └── i18n/               # Fluent 翻译
└── tests/                  # 5 个集成测试
```

---

## 开发

```bash
cargo test
cargo test --test integration_test
RUST_LOG=agentrust=debug cargo run -- task --goal "..."
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

---

## 作者

由 [eating-and-drinking](https://github.com/eating-and-drinking) 创建并维护。

仓库地址：[https://github.com/eating-and-drinking/AgentRust](https://github.com/eating-and-drinking/AgentRust)

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
