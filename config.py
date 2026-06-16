"""
BagBuddy - Configuration
Loads shared public config from .env.public, then user overrides from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_dir = Path(__file__).parent

# 1. Load shared public keys (committed to git)
load_dotenv(dotenv_path=_env_dir / ".env.public", override=False)

# 2. Load user private keys (.env is gitignored, overrides public)
load_dotenv(dotenv_path=_env_dir / ".env", override=True)

# LLM Configuration - user must configure their own API key
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# Langfuse keys (from .env, gitignored - never committed)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://jp.cloud.langfuse.com")


# Proxy configuration (all secret keys are held by the proxy server)
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "").rstrip("/")
# Application
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "9000"))
