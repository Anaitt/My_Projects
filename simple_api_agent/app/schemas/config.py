from pydantic import BaseModel
from typing import Optional

class LLMConfigCreate(BaseModel):
    name: str       # название
    base_url: str   # URL
    api_key: str    # ключ
    model: str      # название модели
    folder_id: Optional[str] = None

class MCPConfigCreate(BaseModel):
    name: str
    url: str
    token: str
