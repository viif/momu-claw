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

## workspace 配置文件说明

`workspace` 下的每个工作区都可以通过一组 Markdown 配置文件定义 Agent 的身份、行为、工具习惯和用户偏好。

### AGENTS.md
用于定义 Agent 的行为准则与工作流，建议填写：
- Agent 名称与使命
- 任务归属与路由边界
- 默认工作流步骤
- 输出要求
- 约束条件

### BOOTSTRAP.md
用于定义首次启动引导，建议填写：
- 文件用途
- 初始化检查项
- 建议的开场提问
- 初始化完成条件

### HEARTBEAT.md
用于定义定时巡检清单，建议填写：
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

## 致谢

感谢 shareAI-lab 提供的 [《claw0》](https://github.com/shareAI-lab/claw0/) 教程。
