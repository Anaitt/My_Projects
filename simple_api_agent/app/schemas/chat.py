from pydantic import BaseModel
from typing import List, Optional


class MessageSchema(BaseModel):
    role: str
    content: str

class ChatCreate(BaseModel):
    name: Optional[str] = "Новый чат"

class UserMessage(BaseModel):
    message: str
