"""
极简 Python Agent 网关

支持：
- 简单工具调用
- 会话持久化: JSONL 保存与恢复
- 上下文保护: tool_result 截断与历史压缩
- 多通道输入输出
- HTTP Webhook Channel
"""

# -------------------------------------------------------------
# 导入
# -------------------------------------------------------------
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
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

# 工作目录 -- 所有文件操作相对于此目录, 防止路径穿越
WORKDIR = Path.cwd()

# 会话目录与上下文保护阈值
SESSIONS_DIR = WORKDIR / ".sessions"
CONTEXT_SAFE_LIMIT = 180000

# HTTP Webhook 配置
HTTP_WEBHOOK_HOST = os.getenv("HTTP_WEBHOOK_HOST", "127.0.0.1")
HTTP_WEBHOOK_PORT = int(os.getenv("HTTP_WEBHOOK_PORT", "50001"))
HTTP_WEBHOOK_PATH = os.getenv("HTTP_WEBHOOK_PATH", "/webhook")


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


def extract_text(blocks: Sequence[object]) -> str:
    text = ""
    for block in blocks:
        if isinstance(block, TextBlock):
            text += block.text
        elif hasattr(block, "text"):
            text += cast(Any, block).text
    return text


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


def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    return f"agent:main:direct:{channel}:{account_id}:{peer_id}"


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
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
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
                    self._send_json(400, {"ok": False, "error": "json body must be an object"})
                    return

                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    self._send_json(400, {"ok": False, "error": "field 'text' is required"})
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
                    channel=channel.name,
                    account_id=channel.account_id,
                    peer_id=peer_id,
                    reply_to=str(payload.get("reply_to") or peer_id),
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
        print_info(f"  HTTP webhook listening on http://{self.host}:{self.port}{self.path}")

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
            target.send(inbound.reply_to or inbound.peer_id, text, **inbound.reply_kwargs)

    def close_all(self) -> None:
        for channel in self.channels.values():
            channel.close()


def receive_cli_input(outbox: queue.Queue[InboundMessage | None], account_id: str) -> None:
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
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "get_current_time": tool_get_current_time,
}


def process_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
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
    """管理 agent 会话的持久化存储。"""

    def __init__(self) -> None:
        self.base_dir = SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "sessions.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()
        self.current_session_id: str | None = None

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

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """从 JSONL 重建 API 格式的 messages[]。"""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        if not self.current_session_id:
            return
        self.append_transcript(
            self.current_session_id,
            {"type": role, "content": content, "ts": time.time()},
        )

    def save_tool_result(
        self,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        result: str,
    ) -> None:
        if not self.current_session_id:
            return
        ts = time.time()
        self.append_transcript(
            self.current_session_id,
            {
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "name": name,
                "input": tool_input,
                "ts": ts,
            },
        )
        self.append_transcript(
            self.current_session_id,
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "ts": ts,
            },
        )

    def append_transcript(self, session_id: str, record: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = datetime.now(
                timezone.utc
            ).isoformat()
            self._index[session_id]["message_count"] += 1
            self._save_index()

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

    def list_sessions(self) -> list[tuple[str, dict[str, Any]]]:
        items = list(self._index.items())
        items.sort(key=lambda item: item[1].get("last_active", ""), reverse=True)
        return items


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


# -------------------------------------------------------------
# ContextGuard -- 上下文溢出保护
# -------------------------------------------------------------
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
        """
        将前 50% 的消息压缩为 LLM 生成的摘要。
        保留最后 N 条消息 (N = max(4, 总数的 20%)) 不变。
        """
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
        """遍历消息列表, 截断过大的 tool_result 块。"""
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
        """
        三阶段重试:
          第0次尝试: 正常调用
          第1次尝试: 截断过大的工具结果
          第2次尝试: 通过 LLM 摘要压缩历史
        """
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
def handle_repl_command(
    command: str,
    store: SessionStore,
    guard: ContextGuard,
    messages: list[dict[str, Any]],
    mgr: ChannelManager,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    处理以 / 开头的命令。
    返回 (是否已处理, messages)。
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"  创建新会话: {sid}" + (f" ({label})" if label else ""))
        return True, []

    if cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("  没有会话记录.")
            return True, messages

        print_info("  会话列表:")
        for sid, meta in sessions:
            active = " <-- current" if sid == store.current_session_id else ""
            label = meta.get("label", "")
            label_str = f" ({label})" if label else ""
            count = meta.get("message_count", 0)
            last = str(meta.get("last_active", "?"))[:19]
            print_info(f"    {sid}{label_str}  msgs={count}  last={last}{active}")
        return True, messages

    if cmd == "/switch":
        if not arg:
            print_warn("  用法: /switch <session_id>")
            return True, messages
        target_id = arg.strip()
        matched = [sid for sid in store._index if sid.startswith(target_id)]
        if len(matched) == 0:
            print_warn(f"  未找到会话: {target_id}")
            return True, messages
        if len(matched) > 1:
            print_warn(f"  前缀不唯一，匹配到: {', '.join(matched)}")
            return True, messages

        sid = matched[0]
        new_messages = store.load_session(sid)
        print_session(f"  切换到会话: {sid} ({len(new_messages)} messages)")
        return True, new_messages

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
        return True, messages

    if cmd == "/compact":
        if len(messages) <= 4:
            print_info("  消息太少，暂不需要压缩 (需 > 4).")
            return True, messages
        print_session("  正在压缩历史...")
        new_messages = guard.compact_history(messages, client, MODEL_ID)
        print_session(f"  {len(messages)} -> {len(new_messages)} messages")
        return True, new_messages

    if cmd == "/channels":
        print_info("  Channels:")
        for name in mgr.list_channels():
            print_info(f"    {name}")
        return True, messages

    if cmd == "/accounts":
        print_info("  Accounts:")
        for account in mgr.accounts:
            token = account.token[:8] + "..." if len(account.token) > 8 else "(none)"
            print_info(f"    {account.channel}/{account.account_id}  token={token}")
        return True, messages

    if cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /channels          List registered channels")
        print_info("    /accounts          List configured accounts")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages

    return False, messages


# -------------------------------------------------------------
# 核心: Agent 回合
# -------------------------------------------------------------
def run_agent_turn(
    inbound: InboundMessage,
    messages: list[dict[str, Any]],
    store: SessionStore,
    guard: ContextGuard,
    mgr: ChannelManager,
) -> None:
    _ = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)

    messages.append({"role": "user", "content": inbound.text})
    store.save_turn("user", inbound.text)

    while True:
        try:
            response = guard.guard_api_call(
                api_client=client,
                model=MODEL_ID,
                system=SYSTEM_PROMPT,
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
        store.save_turn("assistant", serialized_content)

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
                result = process_tool_call(tool_use_block.name, tool_use_block.input)
                store.save_tool_result(
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


# -------------------------------------------------------------
# 核心: Agent 循环
# -------------------------------------------------------------
def agent_loop() -> None:
    store = SessionStore()
    guard = ContextGuard()
    mgr = ChannelManager()

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

    # 恢复最近的会话或创建新会话
    sessions = store.list_sessions()
    if sessions:
        sid = sessions[0][0]
        messages: list[dict[str, Any]] = store.load_session(sid)
        print_session(f"  恢复会话: {sid} ({len(messages)} messages)")
    else:
        sid = store.create_session("initial")
        messages = []
        print_session(f"  创建初始会话: {sid}")

    print_info("=" * 60)
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Session: {store.current_session_id}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info(f"  Channels: {', '.join(mgr.list_channels())}")
    print_info(f"  HTTP Webhook: http://{HTTP_WEBHOOK_HOST}:{HTTP_WEBHOOK_PORT}{HTTP_WEBHOOK_PATH}")
    print_info("  输入 /help 查看命令, 输入 quit 或 exit 退出.")
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
                print(f"\n{DIM}再见.{RESET}")
                break

            user_input = inbound.text
            if inbound.channel == "cli" and user_input.lower() in ("quit", "exit"):
                print(f"{DIM}再见.{RESET}")
                break

            if inbound.channel == "cli" and user_input.startswith("/"):
                handled, messages = handle_repl_command(
                    user_input, store, guard, messages, mgr
                )
                if handled:
                    spawn_cli_reader()
                    continue

            run_agent_turn(inbound, messages, store, guard, mgr)

            if inbound.channel == "cli":
                spawn_cli_reader()
    finally:
        mgr.close_all()


# -------------------------------------------------------------
# 入口
# -------------------------------------------------------------
def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY 未设置.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的 key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
