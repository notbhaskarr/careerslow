"""WebSocket control message types shared between server and browser."""

STOP_PLAYBACK = {"type": "stop_playback"}


def is_control_message(raw: str) -> bool:
    return raw.startswith("{")
