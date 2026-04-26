"""ElevenLabs-related environment variables (shared by API + UI).

Supports a few alternate names so .env matches what users already have.
"""

from __future__ import annotations

import os


def elevenlabs_api_key() -> str | None:
    v = (
        os.getenv("ELEVENLABS_API_KEY")
        or os.getenv("ELEVELABS_API_KEY")
        or os.getenv("elevenlabs_api_key")
        or os.getenv("XI_API_KEY")
    )
    return v.strip() if v else None


def elevenlabs_voice_id() -> str | None:
    v = (
        os.getenv("ELEVENLABS_VOICE_ID")
        or os.getenv("ELEVENLABS_VOICE")
        or os.getenv("elevenlabs_voice_id")
        or os.getenv("VOICE_ID")
        or os.getenv("ELEVENLABS_DEFAULT_VOICE_ID")
        or os.getenv("ELEVENLABS_VOICEID")
    )
    return v.strip() if v else None


def intake_tts_configured() -> bool:
    return bool(elevenlabs_api_key() and elevenlabs_voice_id())
