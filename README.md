# momu-claw
参考 claw0 实现的 Agent 网关

## 安装依赖

```bash
uv sync
```

当前核心依赖：

- `anthropic`
- `python-dotenv`

## 环境变量

程序会自动从 `.env` 加载环境变量。当前需要关注的变量有：

- `ANTHROPIC_API_KEY`
  - 必填，用于身份认证。
- `MODEL_ID`
  - 选填，用于指定调用的模型 ID。
- `ANTHROPIC_BASE_URL`
  - 选填，用于指定自定义 API Base URL。

可参考 `.env.example` 创建本地 `.env` 文件。

```bash
cp .env.example .env
```

## 运行方式

```bash
uv run python agent.py
```

## 致谢

感谢 shareAI-lab 提供的 [《claw0》](https://github.com/shareAI-lab/claw0/) 教程。
