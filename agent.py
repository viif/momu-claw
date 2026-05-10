"""
极简 Python Agent 网关
"""

# -------------------------------------------------------------
# 导入
# -------------------------------------------------------------
import os
import sys
from collections.abc import Sequence

from anthropic import Anthropic
from anthropic.types.message_param import MessageParam
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

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."


# --------------------------------------------------------------
# ANSI 颜色
# --------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def extract_text(blocks: Sequence[object]) -> str:
    text = ""
    for block in blocks:
        if isinstance(block, TextBlock):
            text += block.text
    return text


# -------------------------------------------------------------
# 核心: Agent 循环
# -------------------------------------------------------------
#   1. 收集用户输入, 追加到 messages
#   2. 调用 LLM API
#   3. 检查 stop_reason 决定下一步
# -------------------------------------------------------------


def agent_loop() -> None:
    """主 agent 循环 -- 对话式 REPL."""

    messages: list[MessageParam] = []

    print_info("=" * 60)
    print_info(f"  Model: {MODEL_ID}")
    print_info("  输入 'quit' 或 'exit' 退出. Ctrl+C 同样有效.")
    print_info("=" * 60)
    print()

    while True:
        # --- 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}再见.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- 追加到历史 ---
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # --- 调用 LLM ---
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()
            continue

        # --- 检查 stop_reason ---
        if response.stop_reason == "end_turn":
            assistant_text = extract_text(response.content)
            print_assistant(assistant_text)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

        elif response.stop_reason == "tool_use":
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

        else:
            print_info(f"[stop_reason={response.stop_reason}]")
            assistant_text = extract_text(response.content)
            if assistant_text:
                print_assistant(assistant_text)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )


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
