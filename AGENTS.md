# AGENTS.md

## 常用命令

- 安装依赖：`uv sync`
- 运行程序：`uv run python agent.py`
- 快速语法检查：`uv run python -m py_compile agent.py`

## 项目定位

这是一个学习性质的 Python Agent 网关项目，用于接入 Anthropic 官方 API 或兼容网关。

当前实现仍以 `agent.py` 作为唯一程序入口，并保持单文件代码结构；但功能已经覆盖多 Agent 配置、多工作区、路由绑定、会话持久化、HTTP webhook、可靠投递、记忆、技能、heartbeat 与 cron 后台任务。协作时默认按“小型、单入口、便于直接阅读”的项目处理，不要为了工程化预设多模块拆分。

## 结构说明

- `agent.py`：唯一代码文件，也是程序入口，包含配置、通道、路由、工具、会话、记忆、技能、后台任务与 REPL 命令。
- `agents.json`：多 Agent 配置文件，定义 `id`、`name`、`model`、`dm_scope`。
- `workspace/workspace-<agent_id>/`：每个 Agent 的独立工作区，存放提示词、记忆、技能和定时任务配置。
- `workspace/workspace-<agent_id>/AGENTS.md` 等 Markdown 文件：工作区上下文文件，会在构建系统提示词时加载。
- `workspace/workspace-<agent_id>/skills/*/SKILL.md`：工作区技能目录，使用简单 frontmatter 描述技能元信息。
- `workspace/workspace-<agent_id>/CRON.json`：工作区 cron 定时任务配置。
- `workspace/workspace-<agent_id>/memory/daily/*.jsonl`：运行时写入的每日记忆日志。
- `workspace/delivery-queue/`：运行时磁盘投递队列，失败消息会进入 `failed/`。
- `.sessions/`：运行时 JSONL 会话持久化目录。
- `pyproject.toml`：项目元数据与依赖定义，使用 `uv` 管理。
- `.env.example`：运行所需环境变量示例；新增配置项时要同步更新。
- `uv.lock`：锁定依赖版本；依赖发生变化时应一并更新。

## 修改时的注意事项

- 这是学习项目，优先保持实现直接、清晰、容易从头读懂。
- 虽然功能已经较多，代码仍应尽量维持单文件内的直观组织，避免为少量逻辑引入不必要的抽象或拆分。
- 添加新功能时，要更新 `agent.py` 顶部的功能列表注释，保持文档与代码同步。
- 修改代码时，不要删除原有注释。如果注释过时了，可以在注释前加上 `--- IGNORE ---` 标记，表示这行注释已不再准确，并提醒可以更新它。
- 如果调整模型接入方式、环境变量、默认值、通道配置或 fallback 行为，要同时检查代码中的默认值、`.env.example` 与 `README.md` 是否一致。
- 如果调整工作区上下文文件、技能目录、记忆结构、cron 配置或 agents 配置格式，要同步更新 `README.md` 中的对应说明。
- 如果修改依赖，要同步更新 `pyproject.toml` 与 `uv.lock`，并检查 `README.md` 的依赖列表。
- 当前仓库里没有看到测试、lint 或 formatter 配置；默认使用 `uv run python -m py_compile agent.py` 做快速语法检查。如果后续新增测试或 lint 命令，再把相应命令补充到本文件。

## Git 提交规范

提交信息格式：`type: description`

常用 type：

- `feat`：新功能
- `fix`：缺陷修复
- `refactor`：重构（非功能变更、非缺陷修复）
- `chore`：构建、依赖、CI 等维护性改动
- `style`：代码格式调整（不影响逻辑）
- `docs`：文档变更
