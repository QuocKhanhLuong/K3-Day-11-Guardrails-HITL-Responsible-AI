"""Configuration shared by the Day 11 controlled-agent security lab."""
from __future__ import annotations

from getpass import getpass
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

_PLACEHOLDER_KEYS = {
    "your-google-ai-studio-key-here",
    "your-api-key-here",
    "paste-your-key-here",
    "changeme",
}


def setup_api_key(*, prompt_if_missing: bool = True) -> str:
    """Load and validate ``GOOGLE_API_KEY`` without hanging in CI.

    Environment variables win over ``.env``. Interactive prompting is used only
    when stdin is a terminal; automated runs fail with a clear error instead of
    silently accepting the placeholder from ``.env.example``.
    """
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)

    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key.casefold() in _PLACEHOLDER_KEYS:
        key = ""

    if not key and prompt_if_missing and sys.stdin.isatty():
        key = getpass("GOOGLE_API_KEY: ").strip()

    if not key or key.casefold() in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still a placeholder. "
            "Copy .env.example to .env and add a Google AI Studio key."
        )

    os.environ["GOOGLE_API_KEY"] = key
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")
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
