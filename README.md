# momu-claw

参考 claw0 实现的 Python Agent 网关，用于接入 Anthropic 官方 API 或兼容网关。

当前项目以单文件 `agent.py` 为入口，支持 CLI 与本地 HTTP webhook 两种输入方式，并围绕多 Agent、多工作区、会话持久化、记忆、技能和后台任务做了学习性质的完整串联。

## 功能概览

- 从 `agents.json` 加载多 Agent 配置。
- 每个 Agent 使用独立的 `workspace/workspace-<agent_id>/` 工作区。
- 支持 peer / guild / account / channel / default 五层路由绑定。
- 使用 `.sessions/` 以 JSONL 保存和恢复会话。
- 支持 tool_result 截断与历史压缩，降低上下文溢出风险。
- 支持 API key 轮换、失败冷却、上下文溢出恢复与 fallback model。
- 出站消息先写入 `workspace/delivery-queue/`，后台重试投递。
- 支持 CLI 与 HTTP webhook 输入输出。
- 加载工作区提示词文件、技能、长期记忆与每日记忆日志。
- 支持 heartbeat 后台巡检、`CRON.json` 定时任务与命名 lane 并发调度。

## 环境要求

- Python `>=3.13`
- `uv`

安装依赖：

```bash
uv sync
```

当前核心依赖：

- `anthropic`
- `croniter`
- `python-dotenv`
- `websockets`

## 环境变量

程序会自动从 `.env` 加载环境变量。可参考 `.env.example` 创建本地 `.env` 文件：

```bash
cp .env.example .env
```

当前需要关注的变量有：

- `ANTHROPIC_API_KEY`：必填，主 API key，用于身份认证。
- `ANTHROPIC_API_KEY_BACKUP`：选填，备用 API key，主 key 失败或冷却时轮换使用。
- `ANTHROPIC_API_KEY_EMERGENCY`：选填，应急 API key，备用 key 也不可用时轮换使用。
- `MODEL_ID`：选填，默认模型 ID。未设置时使用 `claude-3-5-sonnet-20241022`。
- `FALLBACK_MODEL_IDS`：选填，fallback 模型链，多个模型用英文逗号分隔。未设置时不启用 fallback。
- `ANTHROPIC_BASE_URL`：选填，自定义 API Base URL，用于接入兼容代理或网关。
- `HTTP_WEBHOOK_HOST`：选填，本地 HTTP webhook 监听地址，默认 `127.0.0.1`。
- `HTTP_WEBHOOK_PORT`：选填，本地 HTTP webhook 监听端口，默认 `50001`。
- `HTTP_WEBHOOK_PATH`：选填，本地 HTTP webhook 路径，默认 `/webhook`。

## 运行方式

```bash
uv run python agent.py
```

启动后程序会：

- 检查 `ANTHROPIC_API_KEY` 是否存在。
- 加载 `agents.json`，如果没有 `main` Agent 会自动注册默认 Agent。
- 启动 CLI 通道与 HTTP webhook 通道。
- 启动磁盘投递队列、heartbeat 与 cron 后台线程。
- 进入交互式 CLI。

快速语法检查：

```bash
uv run python -m py_compile agent.py
```

## HTTP Webhook

默认监听地址为：

```text
http://127.0.0.1:50001/webhook
```

健康检查：

```bash
curl http://127.0.0.1:50001/webhook
```

发送消息：

```bash
curl -X POST http://127.0.0.1:50001/webhook \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","peer_id":"http-user"}'
```

常用入站字段：

- `text`：必填，用户消息文本。
- `peer_id`：选填，对端 ID；未提供时会尝试使用 `sender_id`、`reply_to`，最后回退为 `http-user`。
- `sender_id`：选填，发送者 ID。
- `channel`：选填，通道名；默认 `http`。
- `account_id`：选填，账号 ID；默认 `http-local`。
- `guild_id`：选填，群组或服务器 ID。
- `agent_id`：选填，指定 Agent。
- `session_key`：选填，指定已有会话。
- `reply_to`：选填，回复目标；默认使用 `peer_id`。
- `is_group`：选填，是否群聊。

## CLI 常用命令

启动后可输入 `/help` 查看完整命令。常用命令包括：

- `/new [label]`：重置当前 CLI 会话。
- `/list`：列出会话。
- `/switch <session_key_prefix>`：切换到已有会话。
- `/context`：查看当前上下文估算用量。
- `/compact`：手动压缩历史上下文。
- `/agents`：查看已加载 Agent。
- `/agent <id>`：强制 CLI 使用指定 Agent。
- `/agent off`：恢复路由模式。
- `/bindings`：查看路由绑定。
- `/route <channel> <peer_id> [account_id] [guild_id]`：测试路由解析。
- `/sessions [agent_id]`：查看持久化会话。
- `/channels`：查看已注册通道。
- `/queue`：查看待投递消息。
- `/failed`：查看失败投递。
- `/retry`：将失败投递移回队列。
- `/delivery`：查看投递统计。
- `/stats`：`/delivery` 的别名。
- `/profiles`：查看 API key profile 状态。
- `/cooldowns`：查看冷却中的 profile。
- `/fallback`：查看 fallback 模型链。
- `/resilience`：查看弹性调用统计。
- `/accounts`：查看已配置账号。
- `/heartbeat`：查看当前 Agent 的 heartbeat 状态。
- `/cron`：查看当前 Agent 的 cron 任务。
- `/cron-trigger <job_id>`：手动触发 cron 任务。
- `/lanes`：查看命名 lane 并发状态。
- `/concurrency <lane> <N>`：调整指定 lane 的并发数。
- `/lane-reset`：重置所有 lane 的 generation。
- `/soul`：查看当前 Agent 的 `SOUL.md`。
- `/skills`：查看已发现技能。
- `/memory`：查看记忆统计。
- `/search <query>`：搜索记忆。
- `/prompt`：查看当前系统提示词。
- `/context-files`：查看当前 Agent 加载的工作区上下文文件。

## agents.json 配置文件说明

`agents.json` 位于项目根目录，用于定义可用 Agent 列表。当前字段按下面填写：

- `id`：Agent 的唯一标识，用于内部区分与引用，必须是简短稳定的小写英文、数字、下划线或短横线，例如 `main`、`sage`。
- `name`：展示名称，用于界面或日志展示，例如 `Main`、`Sage`。
- `model`：该 Agent 使用的模型 ID；如果留空字符串 `""`，会沿用 `MODEL_ID`。
- `dm_scope`：私聊上下文隔离范围；当前示例使用 `per-account-channel-peer`，表示按账号 / 频道 / 对端维度隔离会话。

当前示例：

```json
[
  {
    "id": "main",
    "name": "Main",
    "model": "",
    "dm_scope": "per-account-channel-peer"
  },
  {
    "id": "sage",
    "name": "Sage",
    "model": "",
    "dm_scope": "per-account-channel-peer"
  }
]
```

填写建议：

- 只有一个默认 Agent 时，也建议保留数组结构，便于后续扩展。
- `id` 一旦投入使用，后续尽量不要频繁修改，避免影响已有引用、工作区路径或会话映射。
- 如果不同 Agent 需要不同模型，可分别填写各自的 `model`。

## workspace 配置文件说明

每个 Agent 默认使用 `workspace/workspace-<agent_id>/` 作为独立工作区。工具读写相对路径时，默认相对当前 Agent 工作区解析，但最终路径必须位于项目根目录之内。

工作区可以通过一组 Markdown 配置文件定义 Agent 的身份、行为、工具习惯和用户偏好。

### AGENTS.md

用于定义 Agent 的行为准则与工作流，建议填写：

- Agent 名称与使命
- 任务归属与路由边界
- 默认工作流步骤
- 输出要求
- 约束条件

### BOOTSTRAP.md

用于定义首次启动引导。工作区首次使用时，如果尚未生成 `.bootstrap.done`，程序会把该文件内容注入启动对话。建议填写：

- 文件用途
- 初始化检查项
- 建议的开场提问
- 初始化完成条件

### HEARTBEAT.md

用于定义定时巡检清单。heartbeat 默认每隔一段时间检查一次，并只在活跃时间段内触发。建议填写：

- 定时检查任务
- 在什么条件下触发提醒
- 无事发生时返回什么结果
- 哪些情况下应跳过或保持静默

### IDENTITY.md

用于定义 Agent 的身份，建议填写：

- 名称
- 角色
- 专注领域
- 沟通风格
- 优势
- 边界声明

### MEMORY.md

用于存放实际的长期记忆内容，建议填写：

- 已确认且跨会话仍有价值的用户偏好
- 持续有效的协作习惯或角色分工事实
- 对后续决策有长期影响的项目事实

运行时还会把 `memory_write` 写入 `memory/daily/<date>.jsonl`，并通过混合检索为后续对话提供相关记忆。

### SOUL.md

用于定义 Agent 的灵魂与价值观，建议填写：

- 核心原则
- 红线
- 语气风格
- 默认行为准则
- 需要时如何调整表达深度

### TOOLS.md

用于定义工具与环境备忘，建议填写：

- 工具使用原则
- 默认工作习惯
- 环境备注
- 安全约束

### USER.md

用于定义用户画像与协作偏好，建议填写：

- 基础协作信息
- 响应偏好
- 决策风格
- 敏感点与避免事项
- 动态但相对稳定的备注

### CRON.json

用于定义工作区定时任务。程序会周期性读取 `CRON.json`，按 cron 表达式触发任务，并将运行记录写入工作区的 `cron-runs.jsonl`。

建议每个任务包含稳定的 `id`、可读的 `name`、cron 表达式、启用状态和任务内容。实际字段以 `agent.py` 中 `CronService` 当前解析逻辑为准。

### skills/*/SKILL.md

用于定义可按需加载的技能。程序会扫描以下目录：

- `skills/`
- `.skills/`
- `.agents/skills/`

每个技能目录中需要包含 `SKILL.md`，并使用简单 frontmatter 提供：

- `name`：技能名称。
- `description`：技能说明。
- `invocation`：建议触发方式或使用时机。

技能正文默认不会完整注入系统提示词，只会注入技能索引；需要时由 Agent 通过 `load_skill` 工具按名称加载。

## 运行时数据

以下目录由程序运行时创建或更新，通常不需要手动编辑：

- `.sessions/`：会话索引与 JSONL 会话记录。
- `workspace/delivery-queue/`：待投递消息队列与失败队列。
- `workspace/workspace-<agent_id>/memory/daily/`：每日记忆日志。
- `workspace/workspace-<agent_id>/cron-runs.jsonl`：cron 运行记录。
- `workspace/workspace-<agent_id>/.bootstrap.done`：首次引导完成标记。

## 致谢

感谢 shareAI-lab 提供的 [《claw0》](https://github.com/shareAI-lab/claw0/) 教程。
