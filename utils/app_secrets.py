"""Single place the app reads secrets and credentials from.

Lookup order for ``get_secret(name)``:

1. ``st.secrets[name]``  (``.streamlit/secrets.toml`` locally, the app's
   "Secrets" settings on Streamlit Cloud, or a mounted secrets file when
   self-hosted)
2. environment variable ``name``  (also covers the headless MCP server,
   which runs outside Streamlit)
3. ``default``

Nothing sensitive may be hardcoded in the repo. Names in use:

    GEMINI_API_KEY     Gemini LLM calls (recommender, insights, PDF parsing)
    METABASE_API_KEY   Metabase API key (needs Syfe VPN to reach the server)
    BUILDER_PASSWORD   password gate on pages/Builder.py
    SUPABASE_URL       saved Builder portfolios (optional; local JSON fallback)
    SUPABASE_KEY       "

See ``.streamlit/secrets.toml.example`` for a template.
"""
import os
from typing import Optional

import streamlit as st


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the secret ``name`` from st.secrets, then the environment, else ``default``."""
    try:
        value = st.secrets.get(name)
    except Exception:
        # No secrets file present (tests, headless MCP server, fresh checkout).
        value = None
    if value:
        return str(value)
    return os.environ.get(name) or default


def require_secret(name: str) -> str:
    """Like ``get_secret`` but raises a clear error when the secret is not configured."""
    value = get_secret(name)
    if not value:
        raise RuntimeError(
            f"Missing required secret '{name}'. Set it in .streamlit/secrets.toml, "
            f"the Streamlit Cloud app secrets, or as an environment variable."
        )
    return value
