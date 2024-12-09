from enum import Enum


class ChatMessage:
    def __init__(self, content, role):
        self.content = content
        self.role = role
        self.additional_kwargs = {}


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
