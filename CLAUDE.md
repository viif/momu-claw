# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

- 安装依赖：`uv sync`
- 运行程序：`uv run python agent.py`
- 快速语法检查：`uv run python -m py_compile agent.py`

## 项目定位

这是一个学习性质的 Python Agent 网关项目，用于接入 Anthropic 官方 API 或兼容网关。

这个仓库未来预计会一直保持为单文件代码结构，唯一代码文件是 `agent.py`。协作时默认按“小型、单入口、便于直接阅读”的项目处理，不要预设这里会演化成多模块架构。

## 结构说明

- `agent.py`：唯一代码文件，也是程序入口。
- `pyproject.toml`：项目元数据与依赖定义，使用 `uv` 管理。
- `.env.example`：运行所需环境变量示例；新增配置项时要同步更新。
- `uv.lock`：锁定依赖版本；依赖发生变化时应一并更新。

## 修改时的注意事项

- 这是学习项目，优先保持实现直接、清晰、容易从头读懂。
- 因为长期是单文件结构，修改时应控制复杂度，避免为了“工程化”而引入不必要的拆分或抽象。
- 添加新功能时，要更新 `agent.py` 顶部的功能列表注释，保持文档与代码同步。
- 修改代码时，不要删除原有注释。如果注释过时了，可以在注释前加上 `--- IGNORE ---` 标记，表示这行注释已不再准确，并提醒可以更新它。
- 如果调整模型接入方式或配置行为，要同时检查代码中的默认值与 `.env.example` 是否一致。
- 当前仓库里没有看到测试、lint 或 formatter 配置；如果后续新增，再把相应命令补充到本文件。

## Git 提交规范

提交信息格式：`type: description`

常用 type：

- `feat`：新功能
- `fix`：缺陷修复
- `refactor`：重构（非功能变更、非缺陷修复）
- `chore`：构建、依赖、CI 等维护性改动
- `style`：代码格式调整（不影响逻辑）
- `docs`：文档变更
