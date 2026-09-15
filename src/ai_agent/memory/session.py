import dataclasses


@dataclasses.dataclass(slots=True)
class SessionMemory:
    user_id: str
    session_id: str
    messages: list[dict[str, str]] = dataclasses.field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages.clear()
