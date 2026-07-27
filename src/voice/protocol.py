"""WebSocket control message types shared between server and browser."""

STOP_PLAYBACK = {"type": "stop_playback"}


def session_phase(phase: str, detail: str = "") -> dict:
    payload = {"type": "session_phase", "phase": phase}
    if detail:
        payload["detail"] = detail
    return payload


def is_control_message(raw: str) -> bool:
    return raw.startswith("{")
