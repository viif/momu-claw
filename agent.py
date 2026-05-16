"""
极简 Python Agent 网关

支持：
- 从 agents.json 加载多 agent 配置
- 多 agent 独立工作区
- 五层路由绑定: peer / guild / account / channel / default
- 会话持久化: JSONL 保存与恢复
- 上下文保护: tool_result 截断与历史压缩
- 多通道输入输出 (CLI + HTTP webhook)
- 引导加载 / 系统提示词 / 技能 / 记忆 / 混合检索智能
"""

# -------------------------------------------------------------
# 导入
# -------------------------------------------------------------
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import ToolParam
from anthropic.types.text_block import TextBlock
from dotenv import load_dotenv


# --------------------------------------------------------------
# 配置
# --------------------------------------------------------------
load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-3-5-sonnet-20241022")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

# 工具输出最大字符数 -- 防止超大输出撑爆上下文
MAX_TOOL_OUTPUT = 50000

# 工作目录
WORKDIR = Path.cwd()
WORKSPACE_DIR = WORKDIR / "workspace"
AGENTS_CONFIG_PATH = WORKDIR / "agents.json"

# 会话目录与上下文保护阈值
SESSIONS_DIR = WORKDIR / ".sessions"
CONTEXT_SAFE_LIMIT = 180000

# HTTP Webhook 配置
HTTP_WEBHOOK_HOST = os.getenv("HTTP_WEBHOOK_HOST", "127.0.0.1")
HTTP_WEBHOOK_PORT = int(os.getenv("HTTP_WEBHOOK_PORT", "50001"))
HTTP_WEBHOOK_PATH = os.getenv("HTTP_WEBHOOK_PATH", "/webhook")

# Agent / 路由默认配置
DEFAULT_AGENT_ID = "main"
VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")

# Intelligence 配置
BOOTSTRAP_FILES = [
    "SOUL.md",
    "IDENTITY.md",
    "TOOLS.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "AGENTS.md",
    "MEMORY.md",
]
MAX_FILE_CHARS = 20000
MAX_TOTAL_CHARS = 150000
MAX_SKILLS = 150
MAX_SKILLS_PROMPT = 30000


# --------------------------------------------------------------
# ANSI 颜色
# --------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_warn(text: str) -> None:
    print(f"{YELLOW}{text}{RESET}")


def print_session(text: str) -> None:
    print(f"{MAGENTA}{text}{RESET}")


def print_section(title: str) -> None:
    print(f"\n{MAGENTA}{BOLD}--- {title} ---{RESET}")


def extract_text(blocks: Sequence[object]) -> str:
    text = ""
    for block in blocks:
        if isinstance(block, TextBlock):
            text += block.text
        elif hasattr(block, "text"):
            text += cast(Any, block).text
    return text


# -------------------------------------------------------------
# Agent / 路由
# -------------------------------------------------------------
def normalize_agent_id(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    lowered = trimmed.lower()
    if VALID_ID_RE.match(lowered):
        return lowered
    cleaned = INVALID_CHARS_RE.sub("-", lowered).strip("-")[:64]
    return cleaned or DEFAULT_AGENT_ID


# dm_scope 控制私聊隔离粒度:
#   main                      -> agent:{id}:main
#   per-peer                  -> agent:{id}:direct:{peer}
#   per-channel-peer          -> agent:{id}:{ch}:direct:{peer}
#   per-account-channel-peer  -> agent:{id}:{ch}:{acc}:direct:{peer}
def build_session_key(
    agent_id: str,
    channel: str = "",
    account_id: str = "",
    peer_id: str = "",
    dm_scope: str = "per-peer",
) -> str:
    aid = normalize_agent_id(agent_id)
    ch = (channel or "unknown").strip().lower()
    acc = (account_id or "default").strip().lower()
    pid = (peer_id or "").strip().lower()
    if dm_scope == "per-account-channel-peer" and pid:
        return f"agent:{aid}:{ch}:{acc}:direct:{pid}"
    if dm_scope == "per-channel-peer" and pid:
        return f"agent:{aid}:{ch}:direct:{pid}"
    if dm_scope == "per-peer" and pid:
        return f"agent:{aid}:direct:{pid}"
    return f"agent:{aid}:main"


def agent_id_from_session_key(session_key: str) -> str:
    if session_key.startswith("agent:"):
        parts = session_key.split(":", 2)
        if len(parts) >= 2 and parts[1]:
            return normalize_agent_id(parts[1])
    return DEFAULT_AGENT_ID


@dataclass
class Binding:
    agent_id: str
    tier: int
    match_key: str
    match_value: str
    priority: int = 0

    def display(self) -> str:
        names = {1: "peer", 2: "guild", 3: "account", 4: "channel", 5: "default"}
        label = names.get(self.tier, f"tier-{self.tier}")
        return (
            f"[{label}] {self.match_key}={self.match_value} -> "
            f"agent:{self.agent_id} (pri={self.priority})"
        )


class BindingTable:
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        binding.agent_id = normalize_agent_id(binding.agent_id)
        self._bindings.append(binding)
        self._bindings.sort(key=lambda b: (b.tier, -b.priority))

    def remove(self, agent_id: str, match_key: str, match_value: str) -> bool:
        aid = normalize_agent_id(agent_id)
        before = len(self._bindings)
        self._bindings = [
            b
            for b in self._bindings
            if not (
                b.agent_id == aid
                and b.match_key == match_key
                and b.match_value == match_value
            )
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        return list(self._bindings)

    def resolve(
        self,
        channel: str = "",
        account_id: str = "",
        guild_id: str = "",
        peer_id: str = "",
    ) -> tuple[str | None, Binding | None]:
        for binding in self._bindings:
            if binding.tier == 1 and binding.match_key == "peer_id":
                if ":" in binding.match_value:
                    if binding.match_value == f"{channel}:{peer_id}":
                        return binding.agent_id, binding
                elif binding.match_value == peer_id:
                    return binding.agent_id, binding
            elif (
                binding.tier == 2
                and binding.match_key == "guild_id"
                and binding.match_value == guild_id
            ):
                return binding.agent_id, binding
            elif (
                binding.tier == 3
                and binding.match_key == "account_id"
                and binding.match_value == account_id
            ):
                return binding.agent_id, binding
            elif (
                binding.tier == 4
                and binding.match_key == "channel"
                and binding.match_value == channel
            ):
                return binding.agent_id, binding
            elif binding.tier == 5 and binding.match_key == "default":
                return binding.agent_id, binding
        return None, None


@dataclass
class AgentConfig:
    id: str
    name: str
    model: str = ""
    dm_scope: str = "per-peer"
    workspace_dir: str = ""

    @property
    def effective_model(self) -> str:
        return self.model or MODEL_ID


# --------------------------------------------------------------
# Agent 运行时上下文
# --------------------------------------------------------------
@dataclass
class AgentRuntime:
    agent_id: str
    workspace_dir: Path
    bootstrap_data: dict[str, str]
    skills_mgr: "SkillsManager"
    skills_index_block: str
    memory_store: "MemoryStore"


# -------------------------------------------------------------
# 通道
# -------------------------------------------------------------
@dataclass
class InboundMessage:
    """所有通道统一归一化为此结构。"""

    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    guild_id: str = ""
    agent_id: str = ""
    session_key: str = ""
    is_group: bool = False
    reply_to: str = ""
    reply_kwargs: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelAccount:
    """通道账号配置骨架。"""

    channel: str
    account_id: str
    token: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None:
        raise NotImplementedError

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CLIChannel(Channel):
    name = "cli"

    def __init__(self, account: ChannelAccount) -> None:
        self.account_id = account.account_id

    def receive(self) -> InboundMessage | None:
        try:
            text = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None
        return InboundMessage(
            text=text,
            sender_id="cli-user",
            channel=self.name,
            account_id=self.account_id,
            peer_id="cli-user",
            reply_to="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        _ = (to, kwargs)
        print_assistant(text)
        return True


class HTTPWebhookChannel(Channel):
    name = "http"

    def __init__(self, account: ChannelAccount) -> None:
        self.account_id = account.account_id
        self.host = str(account.config.get("host", HTTP_WEBHOOK_HOST))
        self.port = int(account.config.get("port", HTTP_WEBHOOK_PORT))
        self.path = str(account.config.get("path", HTTP_WEBHOOK_PATH))
        self._queue: queue.Queue[InboundMessage] = queue.Queue()
        self._server = ThreadingHTTPServer(
            (self.host, self.port), self._build_handler()
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        channel = self

        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != channel.path:
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return

                content_length = self.headers.get("Content-Length", "0")
                try:
                    body = self.rfile.read(int(content_length))
                except ValueError:
                    self._send_json(400, {"ok": False, "error": "invalid json"})
                    return

                if body.startswith(b"\xef\xbb\xbf"):
                    body = body[3:]

                try:
                    payload = json.loads(body or b"{}")
                except UnicodeDecodeError:
                    try:
                        payload = json.loads((body or b"{}").decode("gb18030"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        self._send_json(400, {"ok": False, "error": "invalid json"})
                        return
                except json.JSONDecodeError:
                    self._send_json(400, {"ok": False, "error": "invalid json"})
                    return

                if not isinstance(payload, dict):
                    self._send_json(
                        400, {"ok": False, "error": "json body must be an object"}
                    )
                    return

                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    self._send_json(
                        400, {"ok": False, "error": "field 'text' is required"}
                    )
                    return

                peer_id = str(
                    payload.get("peer_id")
                    or payload.get("sender_id")
                    or payload.get("reply_to")
                    or "http-user"
                )
                inbound = InboundMessage(
                    text=text.strip(),
                    sender_id=str(payload.get("sender_id") or peer_id),
                    channel=str(payload.get("channel") or channel.name),
                    account_id=str(payload.get("account_id") or channel.account_id),
                    peer_id=peer_id,
                    guild_id=str(payload.get("guild_id") or ""),
                    agent_id=str(payload.get("agent_id") or ""),
                    session_key=str(payload.get("session_key") or ""),
                    reply_to=str(payload.get("reply_to") or peer_id),
                    is_group=bool(payload.get("is_group", False)),
                    raw=payload,
                )
                channel.enqueue(inbound)
                self._send_json(202, {"ok": True, "queued": True})

            def do_GET(self) -> None:
                if self.path == channel.path:
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "channel": channel.name,
                            "path": channel.path,
                            "status": "ready",
                        },
                    )
                    return
                self._send_json(404, {"ok": False, "error": "not found"})

            def log_message(self, format: str, *args: Any) -> None:
                _ = format, args

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return WebhookHandler

    def start(self) -> None:
        self._thread.start()
        print_info(
            f"  HTTP webhook listening on http://{self.host}:{self.port}{self.path}"
        )

    def receive(self) -> InboundMessage | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def enqueue(self, inbound: InboundMessage) -> None:
        self._queue.put(inbound)
        print_info(f"  [http] queued message from {inbound.peer_id}")

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        _ = (to, kwargs)
        print_assistant(text)
        return True

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class ChannelManager:
    def __init__(self) -> None:
        self.channels: dict[str, Channel] = {}
        self.accounts: list[ChannelAccount] = []

    def register(self, channel: Channel) -> None:
        self.channels[channel.name] = channel

    def list_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def broadcast(self, inbound: InboundMessage, text: str) -> None:
        target = self.get(inbound.channel) or self.get("cli")
        if target is not None:
            target.send(
                inbound.reply_to or inbound.peer_id, text, **inbound.reply_kwargs
            )

    def close_all(self) -> None:
        for channel in self.channels.values():
            channel.close()


def receive_cli_input(
    outbox: queue.Queue[InboundMessage | None], account_id: str
) -> None:
    while True:
        try:
            text = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            outbox.put(None)
            return
        if not text:
            continue
        outbox.put(
            InboundMessage(
                text=text,
                sender_id="cli-user",
                channel="cli",
                account_id=account_id,
                peer_id="cli-user",
                reply_to="cli-user",
            )
        )
        return


# -------------------------------------------------------------
# 安全辅助函数
# -------------------------------------------------------------
def safe_path(raw: str) -> Path:
    """
    将用户/模型传入的路径解析为安全的绝对路径.
    防止路径穿越: 最终路径必须在 WORKDIR 之下.
    """
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR.resolve())):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """截断过长的输出, 并附上提示."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


# --------------------------------------------------------------
# Bootstrap 文件加载器
# --------------------------------------------------------------
# 在 agent 启动时加载工作区的 Bootstrap 文件.
# 不同加载模式 (full/minimal/none) 适用于不同场景:
#   full = 主 agent | minimal = 子 agent / cron | none = 最小化
class BootstrapLoader:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    def load_file(self, name: str) -> str:
        path = self.workspace_dir / name
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def truncate_file(self, content: str, max_chars: int = MAX_FILE_CHARS) -> str:
        """截断超长文件内容. 仅保留头部, 在行边界处截断."""
        if len(content) <= max_chars:
            return content
        cut = content.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        return (
            content[:cut]
            + f"\n\n[... truncated ({len(content)} chars total, showing first {cut}) ...]"
        )

    def load_all(self, mode: str = "full") -> dict[str, str]:
        if mode == "none":
            return {}
        names = (
            ["AGENTS.md", "TOOLS.md"] if mode == "minimal" else list(BOOTSTRAP_FILES)
        )
        result: dict[str, str] = {}
        total = 0
        for name in names:
            raw = self.load_file(name)
            if not raw:
                continue
            truncated = self.truncate_file(raw)
            if total + len(truncated) > MAX_TOTAL_CHARS:
                remaining = MAX_TOTAL_CHARS - total
                if remaining > 0:
                    truncated = self.truncate_file(raw, remaining)
                else:
                    break
            result[name] = truncated
            total += len(truncated)
        return result


# --------------------------------------------------------------
# 技能发现与注入
# --------------------------------------------------------------
# 一个技能 = 一个包含 SKILL.md (带 frontmatter) 的目录.
# 按优先级顺序扫描; 同名技能会被后发现的覆盖.
class SkillsManager:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.skills: list[dict[str, str]] = []

    def _parse_frontmatter(self, text: str) -> dict[str, str]:
        """解析简单的 YAML frontmatter, 不依赖 pyyaml."""
        meta: dict[str, str] = {}
        if not text.startswith("---"):
            return meta
        parts = text.split("---", 2)
        if len(parts) < 3:
            return meta
        for line in parts[1].strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            meta[key.strip()] = value.strip()
        return meta

    @staticmethod
    def _normalize_skill_name(value: str) -> str:
        return re.sub(r"\s+", "-", value.strip().lower())

    def _scan_dir(self, base: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        if not base.is_dir():
            return found
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta = self._parse_frontmatter(content)
            if not meta.get("name"):
                continue
            body = ""
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
            found.append(
                {
                    "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "invocation": meta.get("invocation", ""),
                    "body": body,
                    "path": str(child),
                    "lookup_name": self._normalize_skill_name(meta.get("name", "")),
                }
            )
        return found

    def discover(self, extra_dirs: list[Path] | None = None) -> None:
        """按优先级扫描技能目录; 同名技能后者覆盖前者."""
        scan_order: list[Path] = []
        if extra_dirs:
            scan_order.extend(extra_dirs)
        scan_order.append(self.workspace_dir / "skills")
        scan_order.append(self.workspace_dir / ".skills")
        scan_order.append(self.workspace_dir / ".agents" / "skills")

        seen: dict[str, dict[str, str]] = {}
        for directory in scan_order:
            for skill in self._scan_dir(directory):
                seen[skill["name"]] = skill
        self.skills = list(seen.values())[:MAX_SKILLS]

    def format_index_block(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "## Available Skills",
            "",
            "Only the skill catalog is loaded by default.",
            "Use the load_skill tool before relying on a skill's detailed body.",
            "",
        ]
        total = 0
        for skill in self.skills:
            invocation = skill["invocation"] or "(manual)"
            block = (
                f"### Skill: {skill['name']}\n"
                f"Description: {skill['description']}\n"
                f"Invocation: {invocation}\n\n"
            )
            if total + len(block) > MAX_SKILLS_PROMPT:
                lines.append("(... more skills truncated)")
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)

    def get_skill(self, name: str) -> dict[str, str] | None:
        normalized = self._normalize_skill_name(name)
        if not normalized:
            return None
        for skill in self.skills:
            if skill.get("lookup_name") == normalized:
                return skill
        return None

    def load_skill(self, name: str) -> str:
        skill = self.get_skill(name)
        if skill is None:
            available = ", ".join(skill["name"] for skill in self.skills[:20])
            suffix = f" Available skills: {available}" if available else ""
            return f"Error: Skill not found: {name}.{suffix}"
        lines = [
            f"# Skill: {skill['name']}",
            f"Description: {skill['description']}",
            f"Invocation: {skill['invocation'] or '(manual)'}",
            f"Path: {skill['path']}",
            "",
        ]
        if skill.get("body"):
            lines.append(skill["body"])
        else:
            lines.append("(This skill has no body content.)")
        return truncate("\n".join(lines), MAX_SKILLS_PROMPT)


# --------------------------------------------------------------
# 记忆系统
# --------------------------------------------------------------
# 两层存储:
#   MEMORY.md = 长期事实 (手动维护)
#   daily/{date}.jsonl = 每日日志 (通过 agent 工具自动写入)
# 搜索: TF-IDF + 余弦相似度, 纯 Python 实现
class MemoryStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.memory_dir = workspace_dir / "memory" / "daily"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(self, content: str, category: str = "general") -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.memory_dir / f"{today}.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "content": content,
        }
        try:
            with open(path, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return (
                f"Memory saved to {path.relative_to(self.workspace_dir)} ({category})"
            )
        except Exception as exc:
            return f"Error writing memory: {exc}"

    def load_evergreen(self) -> str:
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self) -> list[dict[str, str]]:
        """加载所有记忆并拆分为块 (path + text)."""
        chunks: list[dict[str, str]] = []
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        if self.memory_dir.is_dir():
            for jsonl_file in sorted(self.memory_dir.glob("*.jsonl")):
                try:
                    for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("content", "")
                        if text:
                            category = entry.get("category", "")
                            label = (
                                f"memory/daily/{jsonl_file.name} [{category}]"
                                if category
                                else f"memory/daily/{jsonl_file.name}"
                            )
                            chunks.append({"path": label, "text": text})
                except Exception:
                    continue
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词: 小写英文 + 单个 CJK 字符, 过滤短 token."""
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """TF-IDF + 余弦相似度搜索, 纯 Python 实现."""
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        chunk_tokens = [self._tokenize(chunk["text"]) for chunk in chunks]
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        count = len(chunks)

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            return {
                token: freq * (math.log((count + 1) / (df.get(token, 0) + 1)) + 1)
                for token, freq in tf.items()
            }

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[key] * b[key] for key in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        query_vector = tfidf(query_tokens)
        scored: list[dict[str, Any]] = []
        for index, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(query_vector, tfidf(tokens))
            if score > 0.0:
                snippet = chunks[index]["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                scored.append(
                    {
                        "path": chunks[index]["path"],
                        "score": round(score, 4),
                        "snippet": snippet,
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    # --- Hybrid Memory Search Enhancement ---
    @staticmethod
    def _hash_vector(text: str, dim: int = 64) -> list[float]:
        """使用 hash 投影模拟第二搜索通道, 无需外部向量服务."""
        tokens = MemoryStore._tokenize(text)
        vec = [0.0] * dim
        for token in tokens:
            hashed = hash(token)
            for index in range(dim):
                bit = (hashed >> (index % 62)) & 1
                vec[index] += 1.0 if bit else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _vector_cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    def _vector_search(
        self, query: str, chunks: list[dict[str, str]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """按模拟向量相似度检索."""
        query_vec = self._hash_vector(query)
        scored = []
        for chunk in chunks:
            chunk_vec = self._hash_vector(chunk["text"])
            score = self._vector_cosine(query_vec, chunk_vec)
            if score > 0.0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _keyword_search(
        self, query: str, chunks: list[dict[str, str]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """复用 TF-IDF 检索作为关键词通道, 返回排序结果."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        chunk_tokens = [self._tokenize(chunk["text"]) for chunk in chunks]
        count = len(chunks)
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            return {
                token: freq * (math.log((count + 1) / (df.get(token, 0) + 1)) + 1)
                for token, freq in tf.items()
            }

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[key] * b[key] for key in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        query_vector = tfidf(query_tokens)
        scored = []
        for index, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(query_vector, tfidf(tokens))
            if score > 0.0:
                scored.append({"chunk": chunks[index], "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _merge_hybrid_results(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """融合向量与关键词结果."""
        merged: dict[str, dict[str, Any]] = {}
        for result in vector_results:
            key = result["chunk"]["text"][:100]
            merged[key] = {
                "chunk": result["chunk"],
                "score": result["score"] * vector_weight,
            }
        for result in keyword_results:
            key = result["chunk"]["text"][:100]
            if key in merged:
                merged[key]["score"] += result["score"] * text_weight
            else:
                merged[key] = {
                    "chunk": result["chunk"],
                    "score": result["score"] * text_weight,
                }
        ranked = list(merged.values())
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    @staticmethod
    def _temporal_decay(
        results: list[dict[str, Any]], decay_rate: float = 0.01
    ) -> list[dict[str, Any]]:
        """按日期路径施加时间衰减, 新近记忆更靠前."""
        now = datetime.now(timezone.utc)
        for result in results:
            path = result["chunk"].get("path", "")
            age_days = 0.0
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            if date_match:
                try:
                    chunk_date = datetime.strptime(
                        date_match.group(1), "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                    age_days = (now - chunk_date).total_seconds() / 86400.0
                except ValueError:
                    pass
            result["score"] *= math.exp(-decay_rate * age_days)
        return results

    @staticmethod
    def _mmr_rerank(
        results: list[dict[str, Any]], lambda_param: float = 0.7
    ) -> list[dict[str, Any]]:
        """使用 MMR 去重排, 避免返回过于相似的片段."""
        if len(results) <= 1:
            return results
        tokenized = [MemoryStore._tokenize(r["chunk"]["text"]) for r in results]
        selected: list[int] = []
        remaining = list(range(len(results)))
        reranked: list[dict[str, Any]] = []
        while remaining:
            best_idx = -1
            best_mmr = float("-inf")
            for idx in remaining:
                relevance = results[idx]["score"]
                max_sim = 0.0
                for sel_idx in selected:
                    sim = MemoryStore._jaccard_similarity(
                        tokenized[idx], tokenized[sel_idx]
                    )
                    if sim > max_sim:
                        max_sim = sim
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            reranked.append(results[best_idx])
        return reranked

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """混合检索: keyword -> vector -> merge -> decay -> MMR -> top_k."""
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        keyword_results = self._keyword_search(query, chunks, top_k=10)
        vector_results = self._vector_search(query, chunks, top_k=10)
        merged = self._merge_hybrid_results(vector_results, keyword_results)
        decayed = self._temporal_decay(merged)
        reranked = self._mmr_rerank(decayed)
        results = []
        for result in reranked[:top_k]:
            snippet = result["chunk"]["text"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            results.append(
                {
                    "path": result["chunk"]["path"],
                    "score": round(result["score"], 4),
                    "snippet": snippet,
                }
            )
        return results

    def get_stats(self) -> dict[str, Any]:
        evergreen = self.load_evergreen()
        daily_files = (
            list(self.memory_dir.glob("*.jsonl")) if self.memory_dir.is_dir() else []
        )
        total_entries = 0
        for daily_file in daily_files:
            try:
                total_entries += sum(
                    1
                    for line in daily_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            except Exception:
                pass
        return {
            "evergreen_chars": len(evergreen),
            "daily_files": len(daily_files),
            "daily_entries": total_entries,
        }


# --------------------------------------------------------------
# 系统提示词组装 -- 核心函数
# --------------------------------------------------------------
# 教学演示 8 个关键提示词层级.
# 每轮重建 -- 上一轮可能更新了记忆.
# 模式: full (主 agent) / minimal (子 agent / cron) / none (最小化)
def build_system_prompt(
    agent: AgentConfig,
    runtime: AgentRuntime,
    *,
    loaded_skills: list[str] | None = None,
    memory_context: str = "",
    mode: str = "full",
    channel: str = "cli",
) -> str:
    bootstrap = runtime.bootstrap_data
    sections: list[str] = []

    # 第 1 层: 身份 -- 优先 IDENTITY.md, 否则回退到默认值
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else SYSTEM_PROMPT)

    # 第 2 层: 灵魂 -- 人格注入, 越靠前影响力越强
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # 第 3 层: 工具使用指南
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # 第 4 层: 技能
    if mode == "full" and runtime.skills_index_block:
        sections.append(runtime.skills_index_block)
        if loaded_skills:
            sections.append(
                "## Loaded Skill Context\n\n"
                "The following skill details were loaded explicitly via the load_skill tool in this conversation.\n\n"
                + "\n\n".join(loaded_skills)
            )

    # 第 5 层: 记忆 -- 长期记忆 + 本轮自动搜索结果
    if mode == "full":
        memory_md = bootstrap.get("MEMORY.md", "").strip()
        memory_parts: list[str] = []
        if memory_md:
            memory_parts.append(f"### Evergreen Memory\n\n{memory_md}")
        if memory_context:
            memory_parts.append(
                f"### Recalled Memories (auto-searched)\n\n{memory_context}"
            )
        if memory_parts:
            sections.append("## Memory\n\n" + "\n\n".join(memory_parts))
        sections.append(
            "## Memory Instructions\n\n"
            "- Use memory_write to save important user facts and preferences.\n"
            "- Reference remembered facts naturally in conversation.\n"
            "- Use memory_search to recall specific past information."
        )

    # 第 6 层: Bootstrap 上下文 -- 剩余的 Bootstrap 文件
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # 第 7 层: 运行时上下文
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        "## Runtime Context\n\n"
        f"- Agent ID: {agent.id}\n"
        f"- Agent Name: {agent.name}\n"
        f"- Model: {agent.effective_model}\n"
        f"- Channel: {channel}\n"
        f"- Workspace: {runtime.workspace_dir}\n"
        f"- Current time: {now}\n"
        f"- Prompt mode: {mode}"
    )

    # 第 8 层: 渠道提示
    hints = {
        "cli": "You are responding via a CLI REPL. Markdown is supported.",
        "http": "You are responding via an HTTP webhook bridge. Keep replies concise.",
    }
    sections.append(
        f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}"
    )

    return "\n\n".join(sections)


def _auto_recall(runtime: AgentRuntime, user_message: str) -> str:
    """根据用户消息自动搜索相关记忆, 注入到系统提示词中."""
    results = runtime.memory_store.hybrid_search(user_message, top_k=3)
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)


def _collect_loaded_skills(messages: list[dict[str, Any]]) -> list[str]:
    loaded: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            text = block.get("content", "")
            if not isinstance(text, str) or not text.startswith("# Skill: "):
                continue
            if text not in loaded:
                loaded.append(text)
    return loaded


# -------------------------------------------------------------
# 工具实现
# -------------------------------------------------------------
def tool_bash(command: str, timeout: int = 30) -> str:
    """执行 shell 命令并返回输出."""
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

    print_tool("bash", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKDIR),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += (
                ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
            )
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(file_path: str) -> str:
    """读取文件内容."""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(file_path: str, content: str) -> str:
    """写入内容到文件. 父目录不存在时自动创建."""
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """
    精确替换文件中的文本.
    old_string 必须在文件中恰好出现一次, 否则报错.
    """
    print_tool("edit_file", f"{file_path} (replace {len(old_string)} chars)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return "Error: old_string not found in file. Make sure it matches exactly."
        if count > 1:
            return (
                f"Error: old_string found {count} times. "
                "It must be unique. Provide more surrounding context."
            )

        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_get_current_time() -> str:
    """返回当前 UTC 时间."""
    print_tool("get_current_time", "")
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def tool_memory_write(
    runtime: AgentRuntime | None, content: str, category: str = "general"
) -> str:
    if runtime is None:
        return "Error: No active agent runtime for memory_write"
    print_tool(
        "memory_write", f"agent={runtime.agent_id} [{category}] {content[:60]}..."
    )
    return runtime.memory_store.write_memory(content, category)


def tool_memory_search(runtime: AgentRuntime | None, query: str, top_k: int = 5) -> str:
    if runtime is None:
        return "Error: No active agent runtime for memory_search"
    print_tool("memory_search", f"agent={runtime.agent_id} {query}")
    results = runtime.memory_store.hybrid_search(query, top_k)
    if not results:
        return "No relevant memories found."
    return "\n".join(
        f"[{result['path']}] (score: {result['score']}) {result['snippet']}"
        for result in results
    )


def tool_load_skill(runtime: AgentRuntime | None, name: str) -> str:
    if runtime is None:
        return "Error: No active agent runtime for load_skill"
    print_tool("load_skill", f"agent={runtime.agent_id} {name}")
    return runtime.skills_mgr.load_skill(name)


# -------------------------------------------------------------
# 工具定义: Schema (传给 API) + Handler 调度表
# -------------------------------------------------------------
TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with a new string. "
            "The old_string must appear exactly once in the file. "
            "Always read the file first to get the exact text to replace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in UTC.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "memory_write",
        "description": (
            "Save an important fact or observation to long-term memory. "
            "Use when you learn something worth remembering about the user or context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or observation to remember.",
                },
                "category": {
                    "type": "string",
                    "description": "Category: preference, fact, context, etc.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories for relevant information, ranked by similarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Max results. Default: 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load specialized knowledge by skill name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to load",
                }
            },
            "required": ["name"],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "get_current_time": tool_get_current_time,
}

MEMORY_TOOL_NAMES = {"memory_write", "memory_search", "load_skill"}


def process_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    runtime: AgentRuntime | None = None,
) -> str:
    if tool_name == "memory_write":
        try:
            return tool_memory_write(runtime, **tool_input)
        except TypeError as exc:
            return f"Error: Invalid arguments for {tool_name}: {exc}"
        except Exception as exc:
            return f"Error: {tool_name} failed: {exc}"

    if tool_name == "memory_search":
        try:
            return tool_memory_search(runtime, **tool_input)
        except TypeError as exc:
            return f"Error: Invalid arguments for {tool_name}: {exc}"
        except Exception as exc:
            return f"Error: {tool_name} failed: {exc}"

    if tool_name == "load_skill":
        try:
            return tool_load_skill(runtime, **tool_input)
        except TypeError as exc:
            return f"Error: Invalid arguments for {tool_name}: {exc}"
        except Exception as exc:
            return f"Error: {tool_name} failed: {exc}"

    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"


def is_tool_use_block(block: object) -> bool:
    return getattr(block, "type", None) == "tool_use"


# -------------------------------------------------------------
# SessionStore -- 基于 JSONL 的会话持久化
# -------------------------------------------------------------
class SessionStore:
    """管理多 agent 会话的持久化存储。"""

    def __init__(self) -> None:
        self.base_dir = SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "sessions.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()
        self.current_session_key: str | None = None

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return cast(dict[str, dict[str, Any]], data)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _file_name_for_key(self, session_key: str) -> str:
        digest = hashlib.sha1(session_key.encode("utf-8")).hexdigest()[:16]
        return f"{digest}.jsonl"

    def _session_path(self, session_key: str) -> Path:
        return self.base_dir / self._file_name_for_key(session_key)

    def ensure_session(
        self,
        agent_id: str,
        session_key: str,
        *,
        label: str = "",
        channel: str = "",
        account_id: str = "",
        peer_id: str = "",
        guild_id: str = "",
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        key = session_key
        meta = self._index.get(key)
        if meta is None:
            self._index[key] = {
                "agent_id": normalize_agent_id(agent_id),
                "session_key": session_key,
                "label": label,
                "channel": channel,
                "account_id": account_id,
                "peer_id": peer_id,
                "guild_id": guild_id,
                "created_at": now,
                "last_active": now,
                "message_count": 0,
                "file_name": self._file_name_for_key(session_key),
            }
            self._save_index()
            self._session_path(session_key).touch()
        else:
            updated = False
            if not meta.get("agent_id"):
                meta["agent_id"] = normalize_agent_id(agent_id)
                updated = True
            for field_name, value in {
                "channel": channel,
                "account_id": account_id,
                "peer_id": peer_id,
                "guild_id": guild_id,
            }.items():
                if value and not meta.get(field_name):
                    meta[field_name] = value
                    updated = True
            if updated:
                self._save_index()
        self.current_session_key = session_key
        return session_key

    def load_session(self, session_key: str) -> list[dict[str, Any]]:
        path = self._session_path(session_key)
        if not path.exists():
            return []
        self.current_session_key = session_key
        return self._rebuild_history(path)

    def save_turn(self, session_key: str, role: str, content: Any) -> None:
        self.append_transcript(
            session_key,
            {"type": role, "content": content, "ts": time.time()},
        )

    def save_tool_result(
        self,
        session_key: str,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        result: str,
    ) -> None:
        ts = time.time()
        self.append_transcript(
            session_key,
            {
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "name": name,
                "input": tool_input,
                "ts": ts,
            },
        )
        self.append_transcript(
            session_key,
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "ts": ts,
            },
        )

    def append_transcript(self, session_key: str, record: dict[str, Any]) -> None:
        path = self._session_path(session_key)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = self._index.get(session_key)
        if meta is not None:
            meta["last_active"] = datetime.now(timezone.utc).isoformat()
            meta["message_count"] = int(meta.get("message_count", 0)) + 1
            self._save_index()
        self.current_session_key = session_key

    def _rebuild_history(self, path: Path) -> list[dict[str, Any]]:
        """
        从 JSONL 行重建 API 格式的消息列表。

        Anthropic API 规则决定了这种重建方式:
          - 消息必须 user/assistant 交替
          - tool_use 块属于 assistant 消息
          - tool_result 块属于 user 消息
        """
        messages: list[dict[str, Any]] = []
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == "user":
                messages.append({"role": "user", "content": record["content"]})

            elif rtype == "assistant":
                content = record["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                messages.append({"role": "assistant", "content": content})

            elif rtype == "tool_use":
                block = {
                    "type": "tool_use",
                    "id": record["tool_use_id"],
                    "name": record["name"],
                    "input": record["input"],
                }
                if messages and messages[-1]["role"] == "assistant":
                    content = messages[-1]["content"]
                    if isinstance(content, list):
                        content.append(block)
                    else:
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(content)},
                            block,
                        ]
                else:
                    messages.append({"role": "assistant", "content": [block]})

            elif rtype == "tool_result":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": record["tool_use_id"],
                    "content": record["content"],
                }
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                    and messages[-1]["content"]
                    and isinstance(messages[-1]["content"][0], dict)
                    and messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append({"role": "user", "content": [result_block]})

        return messages

    def list_sessions(self, agent_id: str = "") -> list[tuple[str, dict[str, Any]]]:
        aid = normalize_agent_id(agent_id) if agent_id else ""
        items = list(self._index.items())
        if aid:
            items = [item for item in items if item[1].get("agent_id") == aid]
        items.sort(key=lambda item: item[1].get("last_active", ""), reverse=True)
        return items


class AgentManager:
    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or WORKSPACE_DIR
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._agents: dict[str, AgentConfig] = {}
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._runtimes: dict[str, AgentRuntime] = {}
        self.cli_focus_session_key: str | None = None
        self.cli_forced_agent_id: str = ""

    def register(self, config: AgentConfig) -> AgentConfig:
        aid = normalize_agent_id(config.id)
        workspace_dir = (
            Path(config.workspace_dir)
            if config.workspace_dir
            else self.workspace_root / f"workspace-{aid}"
        )
        workspace_dir.mkdir(parents=True, exist_ok=True)
        normalized = AgentConfig(
            id=aid,
            name=config.name or aid,
            model=config.model,
            dm_scope=config.dm_scope or "per-peer",
            workspace_dir=str(workspace_dir),
        )
        self._agents[aid] = normalized
        loader = BootstrapLoader(workspace_dir)
        bootstrap_data = loader.load_all(mode="full")
        skills_mgr = SkillsManager(workspace_dir)
        skills_mgr.discover()
        memory_store = MemoryStore(workspace_dir)
        self._runtimes[aid] = AgentRuntime(
            agent_id=aid,
            workspace_dir=workspace_dir,
            bootstrap_data=bootstrap_data,
            skills_mgr=skills_mgr,
            skills_index_block=skills_mgr.format_index_block(),
            memory_store=memory_store,
        )
        return normalized

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[AgentConfig]:
        return list(self._agents.values())

    def get_runtime(self, agent_id: str) -> AgentRuntime | None:
        return self._runtimes.get(normalize_agent_id(agent_id))

    def get_session(
        self, session_key: str, store: SessionStore
    ) -> list[dict[str, Any]]:
        if session_key not in self._sessions:
            self._sessions[session_key] = store.load_session(session_key)
        return self._sessions[session_key]

    def set_cli_focus(self, session_key: str) -> None:
        self.cli_focus_session_key = session_key

    def set_cli_forced_agent(self, agent_id: str) -> None:
        self.cli_forced_agent_id = normalize_agent_id(agent_id) if agent_id else ""


def load_agents_config() -> list[AgentConfig]:
    if not AGENTS_CONFIG_PATH.exists():
        return [
            AgentConfig(
                id=DEFAULT_AGENT_ID,
                name="Main",
                model=MODEL_ID,
                dm_scope="per-account-channel-peer",
            )
        ]

    try:
        raw = json.loads(AGENTS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print_warn(
            f"  Failed to read agents.json, falling back to default agent: {exc}"
        )
        return [
            AgentConfig(
                id=DEFAULT_AGENT_ID,
                name="Main",
                model=MODEL_ID,
                dm_scope="per-account-channel-peer",
            )
        ]

    records: list[dict[str, Any]]
    if isinstance(raw, list):
        records = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("agents"), list):
        records = [item for item in raw["agents"] if isinstance(item, dict)]
    else:
        records = []

    agents: list[AgentConfig] = []
    for item in records:
        agents.append(
            AgentConfig(
                id=str(item.get("id") or DEFAULT_AGENT_ID),
                name=str(item.get("name") or item.get("id") or DEFAULT_AGENT_ID),
                model=str(item.get("model") or ""),
                dm_scope=str(item.get("dm_scope") or "per-account-channel-peer"),
                workspace_dir=str(item.get("workspace_dir") or ""),
            )
        )

    if not agents:
        agents.append(
            AgentConfig(
                id=DEFAULT_AGENT_ID,
                name="Main",
                model=MODEL_ID,
                dm_scope="per-account-channel-peer",
            )
        )
    return agents


def load_default_bindings() -> BindingTable:
    table = BindingTable()
    table.add(
        Binding(
            agent_id=DEFAULT_AGENT_ID,
            tier=5,
            match_key="default",
            match_value="*",
            priority=0,
        )
    )
    return table


def resolve_route(
    bindings: BindingTable,
    agent_mgr: AgentManager,
    inbound: InboundMessage,
) -> tuple[str, str, Binding | None]:
    explicit_agent_id = normalize_agent_id(inbound.agent_id) if inbound.agent_id else ""
    matched: Binding | None = None
    if explicit_agent_id:
        agent_id = explicit_agent_id
    else:
        matched_agent_id, matched = bindings.resolve(
            channel=inbound.channel,
            account_id=inbound.account_id,
            guild_id=inbound.guild_id,
            peer_id=inbound.peer_id,
        )
        agent_id = matched_agent_id or DEFAULT_AGENT_ID
    agent = agent_mgr.get_agent(agent_id)
    dm_scope = agent.dm_scope if agent else "per-account-channel-peer"
    session_key = inbound.session_key or build_session_key(
        agent_id,
        channel=inbound.channel,
        account_id=inbound.account_id,
        peer_id=inbound.peer_id,
        dm_scope=dm_scope,
    )
    return agent_id, session_key, matched


# -------------------------------------------------------------
# ContextGuard -- 上下文溢出保护
# -------------------------------------------------------------
def _serialize_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """将消息列表扁平化为纯文本, 用于 LLM 摘要。"""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block['text']}")
                    elif btype == "tool_use":
                        parts.append(
                            f"[{role} called {block.get('name', '?')}]: "
                            f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                        )
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                        parts.append(f"[tool_result]: {preview}")
                elif hasattr(block, "text"):
                    parts.append(f"[{role}]: {cast(Any, block).text}")
    return "\n".join(parts)


class ContextGuard:
    """保护 agent 免受上下文窗口溢出。"""

    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {}), ensure_ascii=False)
                            )
                    else:
                        if hasattr(block, "text"):
                            total += self.estimate_tokens(cast(Any, block).text)
                        elif hasattr(block, "input"):
                            total += self.estimate_tokens(
                                json.dumps(cast(Any, block).input, ensure_ascii=False)
                            )
        return total

    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        """在换行边界处只保留头部进行截断。"""
        max_chars = int(self.max_tokens * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        cut = result.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = result[:cut]
        return (
            head
            + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"
        )

    def compact_history(
        self,
        messages: list[dict[str, Any]],
        api_client: Anthropic,
        model: str,
    ) -> list[dict[str, Any]]:
        total = len(messages)
        if total <= 4:
            return messages

        keep_count = max(4, int(total * 0.2))
        compress_count = max(2, int(total * 0.5))
        compress_count = min(compress_count, total - keep_count)

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]
        old_text = _serialize_messages_for_summary(old_messages)

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            summary_resp = api_client.messages.create(
                model=model,
                max_tokens=2048,
                system="You are a conversation summarizer. Be concise and factual.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_text = extract_text(summary_resp.content)
            print_session(
                f"  [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            print_warn(f"  [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Understood, I have the context from our previous conversation.",
                    }
                ],
            },
        ]
        compacted.extend(recent_messages)
        return compacted

    def _truncate_large_tool_results(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and isinstance(block.get("content"), str)
                    ):
                        block = dict(block)
                        block["content"] = self.truncate_tool_result(block["content"])
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    def guard_api_call(
        self,
        api_client: Anthropic,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolParam] | None = None,
        max_retries: int = 2,
    ) -> Any:
        current_messages = messages

        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 8096,
                    "system": system,
                    "messages": current_messages,
                }
                if tools:
                    kwargs["tools"] = tools
                result = api_client.messages.create(**kwargs)
                if current_messages is not messages:
                    messages.clear()
                    messages.extend(current_messages)
                return result
            except Exception as exc:
                error_str = str(exc).lower()
                is_overflow = "context" in error_str or "token" in error_str

                if not is_overflow or attempt >= max_retries:
                    raise

                if attempt == 0:
                    print_warn(
                        "  [guard] Context overflow detected, truncating large tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(
                        current_messages
                    )
                elif attempt == 1:
                    print_warn(
                        "  [guard] Still overflowing, compacting conversation history..."
                    )
                    current_messages = self.compact_history(
                        current_messages, api_client, model
                    )

        raise RuntimeError("guard_api_call: exhausted retries")


# -------------------------------------------------------------
# REPL 命令
# -------------------------------------------------------------
def cmd_bindings(bindings: BindingTable) -> None:
    all_bindings = bindings.list_all()
    if not all_bindings:
        print_info("  (no bindings)")
        return
    print_info(f"  Route Bindings ({len(all_bindings)}):")
    colors = [MAGENTA, BLUE, CYAN, GREEN, DIM]
    for binding in all_bindings:
        color = colors[min(binding.tier - 1, 4)]
        print(f"  {color}{binding.display()}{RESET}")


def cmd_route(bindings: BindingTable, agents: AgentManager, args: str) -> None:
    parts = args.strip().split()
    if len(parts) < 2:
        print_warn("  Usage: /route <channel> <peer_id> [account_id] [guild_id]")
        return
    inbound = InboundMessage(
        text="",
        sender_id=parts[1],
        channel=parts[0],
        peer_id=parts[1],
        account_id=parts[2] if len(parts) > 2 else "",
        guild_id=parts[3] if len(parts) > 3 else "",
    )
    agent_id, session_key, matched = resolve_route(bindings, agents, inbound)
    agent = agents.get_agent(agent_id)
    print_info("  Route Resolution:")
    print_info(
        f"    input: ch={inbound.channel} peer={inbound.peer_id} "
        f"acc={inbound.account_id or '-'} guild={inbound.guild_id or '-'}"
    )
    print_info(f"    binding: {matched.display() if matched else 'default fallback'}")
    print_info(f"    agent: {agent_id} ({agent.name if agent else '?'})")
    print_info(f"    session: {session_key}")


def cmd_agents(agent_mgr: AgentManager) -> None:
    agents = agent_mgr.list_agents()
    if not agents:
        print_info("  (no agents)")
        return
    print_info(f"  Agents ({len(agents)}):")
    for agent in agents:
        runtime = agent_mgr.get_runtime(agent.id)
        memory_stats = runtime.memory_store.get_stats() if runtime else {}
        bootstrap_count = len(runtime.bootstrap_data) if runtime else 0
        skills_count = len(runtime.skills_mgr.skills) if runtime else 0
        print_info(
            f"    {agent.id} ({agent.name}) model={agent.effective_model} "
            f"dm_scope={agent.dm_scope} workspace={agent.workspace_dir}"
        )
        print_info(
            f"      bootstrap={bootstrap_count} skills={skills_count} "
            f"memory={memory_stats.get('evergreen_chars', 0)}c/"
            f"{memory_stats.get('daily_files', 0)}f/"
            f"{memory_stats.get('daily_entries', 0)}e"
        )


def cmd_sessions(
    store: SessionStore, agent_id: str = "", current_session_key: str = ""
) -> None:
    sessions = store.list_sessions(agent_id)
    if not sessions:
        print_info("  No session records.")
        return
    print_info(f"  Sessions ({len(sessions)}):")
    for session_key, meta in sessions:
        label = meta.get("label", "")
        label_str = f" ({label})" if label else ""
        count = meta.get("message_count", 0)
        last = str(meta.get("last_active", "?"))[:19]
        aid = meta.get("agent_id", "?")
        active = " <-- current" if session_key == current_session_key else ""
        print_info(
            f"    {session_key}{label_str}  agent={aid} msgs={count} last={last}{active}"
        )


def current_cli_runtime(agent_mgr: AgentManager) -> AgentRuntime | None:
    if agent_mgr.cli_forced_agent_id:
        return agent_mgr.get_runtime(agent_mgr.cli_forced_agent_id)
    if agent_mgr.cli_focus_session_key:
        return agent_mgr.get_runtime(
            agent_id_from_session_key(agent_mgr.cli_focus_session_key)
        )
    return agent_mgr.get_runtime(DEFAULT_AGENT_ID)


def handle_repl_command(
    command: str,
    store: SessionStore,
    guard: ContextGuard,
    messages: list[dict[str, Any]],
    mgr: ChannelManager,
    agent_mgr: AgentManager,
    bindings: BindingTable,
) -> tuple[bool, list[dict[str, Any]], str | None]:
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        target_agent_id = agent_mgr.cli_forced_agent_id
        if not target_agent_id:
            focus_key = agent_mgr.cli_focus_session_key or ""
            if focus_key.startswith("agent:"):
                key_parts = focus_key.split(":", 2)
                if len(key_parts) >= 2 and key_parts[1]:
                    target_agent_id = key_parts[1]
        target_agent_id = normalize_agent_id(target_agent_id or DEFAULT_AGENT_ID)
        agent = agent_mgr.get_agent(target_agent_id)
        if agent is None:
            print_warn(f"  Agent not found: {target_agent_id}")
            return True, messages, None

        new_session_key = build_session_key(
            target_agent_id,
            channel="cli",
            account_id="cli-local",
            peer_id="cli-user",
            dm_scope=agent.dm_scope,
        )
        store.ensure_session(
            target_agent_id,
            new_session_key,
            label=arg,
            channel="cli",
            account_id="cli-local",
            peer_id="cli-user",
        )
        agent_mgr.set_cli_focus(new_session_key)
        store.current_session_key = new_session_key
        print_session(
            f"  Reset current CLI session: {new_session_key}"
            + (f" ({arg})" if arg else "")
        )
        return True, [], new_session_key

    if cmd == "/list":
        cmd_sessions(store, current_session_key=agent_mgr.cli_focus_session_key or "")
        return True, messages, None

    if cmd == "/switch":
        if not arg:
            print_warn("  Usage: /switch <session_key_prefix>")
            return True, messages, None
        target = arg.strip()
        matched = [key for key in store._index if key.startswith(target)]
        if len(matched) == 0:
            print_warn(f"  Session not found: {target}")
            return True, messages, None
        if len(matched) > 1:
            print_warn(f"  Prefix is not unique, matched: {', '.join(matched)}")
            return True, messages, None
        session_key = matched[0]
        new_messages = store.load_session(session_key)
        agent_mgr.set_cli_focus(session_key)
        store.current_session_key = session_key
        print_session(
            f"  Switched to session: {session_key} ({len(new_messages)} messages)"
        )
        return True, new_messages, session_key

    if cmd == "/context":
        estimated = guard.estimate_messages_tokens(messages)
        pct = (estimated / guard.max_tokens) * 100 if guard.max_tokens else 0
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        color = GREEN if pct < 50 else (YELLOW if pct < 80 else RED)
        print_info(f"  Context usage: ~{estimated:,} / {guard.max_tokens:,} tokens")
        print(f"  {color}[{bar}] {pct:.1f}%{RESET}")
        print_info(f"  Messages: {len(messages)}")
        return True, messages, None

    if cmd == "/compact":
        if len(messages) <= 4:
            print_info("  Too few messages to compact yet (need > 4).")
            return True, messages, None
        print_session("  Compacting history...")
        new_messages = guard.compact_history(messages, client, MODEL_ID)
        print_session(f"  {len(messages)} -> {len(new_messages)} messages")
        return True, new_messages, None

    if cmd == "/channels":
        print_info("  Channels:")
        for name in mgr.list_channels():
            print_info(f"    {name}")
        return True, messages, None

    if cmd == "/accounts":
        print_info("  Accounts:")
        for account in mgr.accounts:
            token = account.token[:8] + "..." if len(account.token) > 8 else "(none)"
            print_info(f"    {account.channel}/{account.account_id}  token={token}")
        return True, messages, None

    if cmd == "/agents":
        cmd_agents(agent_mgr)
        return True, messages, None

    if cmd == "/bindings":
        cmd_bindings(bindings)
        return True, messages, None

    if cmd == "/route":
        cmd_route(bindings, agent_mgr, arg)
        return True, messages, None

    if cmd == "/sessions":
        cmd_sessions(store, arg, agent_mgr.cli_focus_session_key or "")
        return True, messages, None

    if cmd == "/agent":
        if not arg:
            print_info(f"  force={agent_mgr.cli_forced_agent_id or '(off)'}")
            return True, messages, None
        if arg.lower() == "off":
            agent_mgr.set_cli_forced_agent("")
            routed_agent_id, routed_session_key, _ = resolve_route(
                bindings,
                agent_mgr,
                InboundMessage(
                    text="",
                    sender_id="cli-user",
                    channel="cli",
                    account_id="cli-local",
                    peer_id="cli-user",
                    reply_to="cli-user",
                ),
            )
            store.ensure_session(
                routed_agent_id,
                routed_session_key,
                label="cli",
                channel="cli",
                account_id="cli-local",
                peer_id="cli-user",
            )
            new_messages = agent_mgr.get_session(routed_session_key, store)
            agent_mgr.set_cli_focus(routed_session_key)
            store.current_session_key = routed_session_key
            print_info(f"  Routing mode restored. focus={routed_session_key}")
            return True, new_messages, routed_session_key
        aid = normalize_agent_id(arg)
        if agent_mgr.get_agent(aid):
            agent_mgr.set_cli_forced_agent(aid)
            print_info(f"  Forcing CLI agent: {aid}")
        else:
            print_warn(f"  Agent not found: {aid}")
        return True, messages, None

    runtime = current_cli_runtime(agent_mgr)

    if cmd == "/soul":
        print_section("SOUL.md")
        soul = runtime.bootstrap_data.get("SOUL.md", "") if runtime else ""
        print(soul if soul else f"{DIM}(SOUL.md not found){RESET}")
        return True, messages, None

    if cmd == "/skills":
        print_section("Discovered Skills")
        skills = runtime.skills_mgr.skills if runtime else []
        if not skills:
            print(f"{DIM}(No skills found){RESET}")
        else:
            for skill in skills:
                print(
                    f"  {BLUE}{skill['invocation'] or '(manual)'}{RESET}  "
                    f"{skill['name']} - {skill['description']}"
                )
                print(f"    {DIM}path: {skill['path']}{RESET}")
        return True, messages, None

    if cmd == "/memory":
        print_section("Memory Stats")
        if runtime is None:
            print(f"{DIM}(No current agent runtime){RESET}")
            return True, messages, None
        stats = runtime.memory_store.get_stats()
        print(f"  Agent: {runtime.agent_id}")
        print(f"  Evergreen memory (MEMORY.md): {stats['evergreen_chars']} chars")
        print(f"  Daily files: {stats['daily_files']}")
        print(f"  Daily entries: {stats['daily_entries']}")
        return True, messages, None

    if cmd == "/search":
        if not arg:
            print_warn("  Usage: /search <query>")
            return True, messages, None
        print_section(f"Memory Search: {arg}")
        if runtime is None:
            print(f"{DIM}(No current agent runtime){RESET}")
            return True, messages, None
        results = runtime.memory_store.hybrid_search(arg)
        if not results:
            print(f"{DIM}(No results){RESET}")
        else:
            for result in results:
                color = GREEN if result["score"] > 0.3 else DIM
                print(f"  {color}[{result['score']:.4f}]{RESET} {result['path']}")
                print(f"    {result['snippet']}")
        return True, messages, None

    if cmd == "/prompt":
        print_section("Full System Prompt")
        if runtime is None:
            print(f"{DIM}(No current agent runtime){RESET}")
            return True, messages, None
        focus_key = agent_mgr.cli_focus_session_key or ""
        channel_name = "cli"
        prompt = build_system_prompt(
            agent_mgr.get_agent(runtime.agent_id)
            or AgentConfig(runtime.agent_id, runtime.agent_id),
            runtime,
            memory_context=_auto_recall(runtime, "show prompt"),
            channel=channel_name if focus_key else "cli",
        )
        if len(prompt) > 3000:
            print(prompt[:3000])
            print(
                f"\n{DIM}... ({len(prompt) - 3000} more chars, total {len(prompt)}){RESET}"
            )
        else:
            print(prompt)
        print(f"\n{DIM}Total prompt length: {len(prompt)} chars{RESET}")
        return True, messages, None

    if cmd == "/bootstrap":
        print_section("Bootstrap Files")
        if runtime is None or not runtime.bootstrap_data:
            print(f"{DIM}(No Bootstrap files loaded){RESET}")
        else:
            for name, content in runtime.bootstrap_data.items():
                print(f"  {BLUE}{name}{RESET}: {len(content)} chars")
        total = sum(
            len(value) for value in (runtime.bootstrap_data.values() if runtime else [])
        )
        print(f"\n  {DIM}Total: {total} chars (limit: {MAX_TOTAL_CHARS}){RESET}")
        return True, messages, None

    if cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Reset current CLI session")
        print_info("    /list              List all sessions")
        print_info("    /switch <prefix>   Switch CLI focus session")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /channels          List registered channels")
        print_info("    /accounts          List configured accounts")
        print_info("    /agents            List registered agents")
        print_info("    /bindings          List route bindings")
        print_info("    /route ...         Preview route resolution")
        print_info("    /sessions [agent]  List sessions")
        print_info("    /agent <id|off>    Force CLI agent")
        print_info("    /soul              Show current agent SOUL.md")
        print_info("    /skills            List current agent skills")
        print_info("    /memory            Show current agent memory stats")
        print_info("    /search <query>    Search current agent memory")
        print_info("    /prompt            Show current agent system prompt")
        print_info("    /bootstrap         Show current agent bootstrap files")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages, None

    return False, messages, None


# -------------------------------------------------------------
# 核心: Agent 回合
# -------------------------------------------------------------
def run_agent_session_turn(
    inbound: InboundMessage,
    agent: AgentConfig,
    runtime: AgentRuntime,
    session_key: str,
    messages: list[dict[str, Any]],
    store: SessionStore,
    guard: ContextGuard,
    mgr: ChannelManager,
) -> None:
    messages.append({"role": "user", "content": inbound.text})
    store.save_turn(session_key, "user", inbound.text)

    while True:
        memory_context = _auto_recall(runtime, inbound.text)
        if memory_context:
            print_info(f"  [memory] auto recall hit for agent:{runtime.agent_id}")
        loaded_skills = _collect_loaded_skills(messages)
        system_prompt = build_system_prompt(
            agent,
            runtime,
            loaded_skills=loaded_skills,
            memory_context=memory_context,
            channel=inbound.channel or "cli",
        )

        try:
            response = guard.guard_api_call(
                api_client=client,
                model=agent.effective_model,
                system=system_prompt,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return

        messages.append({"role": "assistant", "content": response.content})

        serialized_content = []
        for block in response.content:
            if hasattr(block, "text"):
                serialized_content.append(
                    {"type": "text", "text": cast(Any, block).text}
                )
            elif is_tool_use_block(block):
                tool_use_block = cast(Any, block)
                serialized_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_use_block.id,
                        "name": tool_use_block.name,
                        "input": tool_use_block.input,
                    }
                )
        store.save_turn(session_key, "assistant", serialized_content)

        if response.stop_reason == "end_turn":
            assistant_text = extract_text(response.content)
            if assistant_text:
                mgr.broadcast(inbound, assistant_text)
            return

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if not is_tool_use_block(block):
                    continue

                tool_use_block = cast(Any, block)
                result = process_tool_call(
                    tool_use_block.name,
                    tool_use_block.input,
                    runtime=runtime,
                )
                store.save_tool_result(
                    session_key,
                    tool_use_block.id,
                    tool_use_block.name,
                    tool_use_block.input,
                    result,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            continue

        print_info(f"[stop_reason={response.stop_reason}]")
        assistant_text = extract_text(response.content)
        if assistant_text:
            mgr.broadcast(inbound, assistant_text)
        return


def dispatch_inbound(
    inbound: InboundMessage,
    store: SessionStore,
    guard: ContextGuard,
    mgr: ChannelManager,
    agent_mgr: AgentManager,
    bindings: BindingTable,
) -> None:
    if inbound.channel == "cli":
        forced_agent_id = agent_mgr.cli_forced_agent_id or inbound.agent_id
        focus_session_key = agent_mgr.cli_focus_session_key or ""
        session_key = inbound.session_key
        if forced_agent_id:
            expected_prefix = f"agent:{normalize_agent_id(forced_agent_id)}:"
            if focus_session_key.startswith(expected_prefix):
                session_key = focus_session_key
        else:
            session_key = focus_session_key or inbound.session_key
        inbound = InboundMessage(
            text=inbound.text,
            sender_id=inbound.sender_id,
            channel=inbound.channel,
            account_id=inbound.account_id,
            peer_id=inbound.peer_id,
            guild_id=inbound.guild_id,
            agent_id=forced_agent_id,
            session_key=session_key,
            is_group=inbound.is_group,
            reply_to=inbound.reply_to,
            reply_kwargs=dict(inbound.reply_kwargs),
            raw=dict(inbound.raw),
        )

    agent_id, session_key, matched = resolve_route(bindings, agent_mgr, inbound)
    agent = agent_mgr.get_agent(agent_id)
    runtime = agent_mgr.get_runtime(agent_id)
    if agent is None:
        print_warn(f"  Agent not found: {agent_id}")
        return
    if runtime is None:
        print_warn(f"  Agent runtime not found: {agent_id}")
        return

    store.ensure_session(
        agent_id,
        session_key,
        channel=inbound.channel,
        account_id=inbound.account_id,
        peer_id=inbound.peer_id,
        guild_id=inbound.guild_id,
    )
    messages = agent_mgr.get_session(session_key, store)
    if inbound.channel == "cli":
        agent_mgr.set_cli_focus(session_key)

    if matched is not None:
        print_info(f"  [route] {matched.display()}")
    else:
        print_info(f"  [route] explicit/default -> agent:{agent_id}")
    print_info(
        f"  [session] agent={agent_id} workspace={agent.workspace_dir} key={session_key}"
    )

    run_agent_session_turn(
        inbound,
        agent,
        runtime,
        session_key,
        messages,
        store,
        guard,
        mgr,
    )


# -------------------------------------------------------------
# 核心: Agent 循环
# -------------------------------------------------------------
def agent_loop() -> None:
    store = SessionStore()
    guard = ContextGuard()
    mgr = ChannelManager()
    agent_mgr = AgentManager()

    for config in load_agents_config():
        agent_mgr.register(config)

    if not agent_mgr.get_agent(DEFAULT_AGENT_ID):
        agent_mgr.register(
            AgentConfig(
                id=DEFAULT_AGENT_ID,
                name="Main",
                model=MODEL_ID,
                dm_scope="per-account-channel-peer",
            )
        )

    bindings = load_default_bindings()

    cli_account = ChannelAccount(channel="cli", account_id="cli-local")
    mgr.accounts.append(cli_account)
    cli = CLIChannel(cli_account)
    mgr.register(cli)

    http_account = ChannelAccount(
        channel="http",
        account_id="http-local",
        config={
            "host": HTTP_WEBHOOK_HOST,
            "port": HTTP_WEBHOOK_PORT,
            "path": HTTP_WEBHOOK_PATH,
        },
    )
    mgr.accounts.append(http_account)
    http_channel = HTTPWebhookChannel(http_account)
    mgr.register(http_channel)
    http_channel.start()

    cli_queue: queue.Queue[InboundMessage | None] = queue.Queue()
    cli_reader_thread: threading.Thread | None = None

    def spawn_cli_reader() -> None:
        nonlocal cli_reader_thread
        if cli_reader_thread is not None and cli_reader_thread.is_alive():
            return

        def run_once() -> None:
            nonlocal cli_reader_thread
            try:
                receive_cli_input(cli_queue, cli_account.account_id)
            finally:
                cli_reader_thread = None

        cli_reader_thread = threading.Thread(target=run_once, daemon=True)
        cli_reader_thread.start()

    cli_default_agent = (
        agent_mgr.get_agent(DEFAULT_AGENT_ID) or agent_mgr.list_agents()[0]
    )
    cli_session_key = build_session_key(
        cli_default_agent.id,
        channel="cli",
        account_id=cli_account.account_id,
        peer_id="cli-user",
        dm_scope=cli_default_agent.dm_scope,
    )
    store.ensure_session(
        cli_default_agent.id,
        cli_session_key,
        label="cli",
        channel="cli",
        account_id=cli_account.account_id,
        peer_id="cli-user",
    )
    cli_messages = agent_mgr.get_session(cli_session_key, store)
    agent_mgr.set_cli_focus(cli_session_key)
    print_session(f"  CLI session: {cli_session_key} ({len(cli_messages)} messages)")

    print_info("=" * 60)
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Agents config: {AGENTS_CONFIG_PATH.name}")
    print_info(f"  Agents loaded: {len(agent_mgr.list_agents())}")
    for agent in agent_mgr.list_agents():
        runtime = agent_mgr.get_runtime(agent.id)
        stats = (
            runtime.memory_store.get_stats()
            if runtime
            else {
                "evergreen_chars": 0,
                "daily_files": 0,
                "daily_entries": 0,
            }
        )
        print_info(f"    - {agent.id}: workspace={agent.workspace_dir}")
        print_info(
            f"      bootstrap={len(runtime.bootstrap_data) if runtime else 0} "
            f"skills={len(runtime.skills_mgr.skills) if runtime else 0} "
            f"memory={stats['evergreen_chars']}c/{stats['daily_files']}f/{stats['daily_entries']}e"
        )
    print_info(f"  Bindings: {len(bindings.list_all())}")
    print_info(
        f"  Tools: {', '.join(list(TOOL_HANDLERS.keys()) + sorted(MEMORY_TOOL_NAMES))}"
    )
    print_info(f"  Channels: {', '.join(mgr.list_channels())}")
    print_info(
        f"  HTTP Webhook: http://{HTTP_WEBHOOK_HOST}:{HTTP_WEBHOOK_PORT}{HTTP_WEBHOOK_PATH}"
    )
    print_info("  Enter /help to view commands, or quit / exit to leave.")
    print_info("=" * 60)
    print()

    spawn_cli_reader()

    try:
        while True:
            inbound = http_channel.receive()
            if inbound is None:
                try:
                    inbound = cli_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

            if inbound is None:
                print(f"\n{DIM}Goodbye.{RESET}")
                break

            user_input = inbound.text
            if inbound.channel == "cli" and user_input.lower() in ("quit", "exit"):
                print(f"{DIM}Goodbye.{RESET}")
                break

            if inbound.channel == "cli" and user_input.startswith("/"):
                focus_key = agent_mgr.cli_focus_session_key or cli_session_key
                focus_messages = agent_mgr.get_session(focus_key, store)
                handled, new_messages, new_focus_key = handle_repl_command(
                    user_input,
                    store,
                    guard,
                    focus_messages,
                    mgr,
                    agent_mgr,
                    bindings,
                )
                if handled:
                    target_focus_key = new_focus_key or focus_key
                    agent_mgr._sessions[target_focus_key] = new_messages
                    if new_focus_key and new_focus_key != focus_key:
                        agent_mgr._sessions.pop(focus_key, None)
                    spawn_cli_reader()
                    continue

            dispatch_inbound(inbound, store, guard, mgr, agent_mgr, bindings)

            if inbound.channel == "cli":
                spawn_cli_reader()
    finally:
        mgr.close_all()


# -------------------------------------------------------------
# 入口
# -------------------------------------------------------------
def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY is not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
