"""Accumulate STT fragments into one user turn before committing to the LLM."""


class UtteranceBuffer:
    def __init__(self):
        self._parts: list[str] = []
        self.user_speaking = False

    def append(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned:
            self._parts.append(cleaned)
        return self.joined()

    def joined(self) -> str:
        return " ".join(self._parts)

    def word_count(self) -> int:
        joined = self.joined().strip()
        if not joined:
            return 0
        return len(joined.split())

    def has_content(self) -> bool:
        return bool(self.joined().strip())

    def clear(self):
        self._parts.clear()
