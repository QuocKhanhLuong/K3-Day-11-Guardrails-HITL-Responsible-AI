"""Configuration shared by the Day 11 controlled-agent security lab."""
from __future__ import annotations

from getpass import getpass
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

_PLACEHOLDER_KEYS = {
    "your-openai-api-key-here",
    "your-api-key-here",
    "paste-your-key-here",
    "changeme",
}


def setup_api_key(*, prompt_if_missing: bool = True) -> str:
    """Load and validate ``OPENAI_API_KEY`` without hanging in CI."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key.casefold() in _PLACEHOLDER_KEYS:
        key = ""

    if not key and prompt_if_missing and sys.stdin.isatty():
        key = getpass("OPENAI_API_KEY: ").strip()

    if not key or key.casefold() in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still a placeholder. "
            "Copy .env.example to .env and add an OpenAI API key."
        )

    os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("OPENAI_MODEL", "gpt-4.1-mini")
    os.environ.setdefault("OPENAI_JUDGE_MODEL", os.environ["OPENAI_MODEL"])
    return key


ALLOWED_TOPICS = (
    "banking", "bank", "account", "transaction", "transfer", "loan",
    "interest", "interest rate", "savings", "credit", "credit card",
    "deposit", "withdrawal", "balance", "payment", "joint account", "atm",
    "tai khoan", "tài khoản", "giao dich", "giao dịch", "tiet kiem",
    "tiết kiệm", "lai suat", "lãi suất", "chuyen tien", "chuyển tiền",
    "the tin dung", "thẻ tín dụng", "so du", "số dư", "vay", "ngan hang",
    "ngân hàng",
)

BLOCKED_TOPICS = (
    "hack", "exploit", "weapon", "drug", "illegal", "violence",
    "gambling", "bomb", "kill", "steal", "malware", "ransomware",
)
