import uuid
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.auth import verify_auth
from app.core.agent import run_agent
from app.db.base import init_db, get_db
from app.db.models import LLMConfig, MCPConfig, Chat, Message
from app.schemas.chat import ChatCreate, UserMessage
from app.schemas.config import LLMConfigCreate, MCPConfigCreate



app = FastAPI(title="Simple Agent API")

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Simple Agent API is running"}

@app.get("/secure-data")
def read_data(user_id: str = Depends(verify_auth)):
    return {"message": f"Hello {user_id}! This is your personal data!"}

@app.post("/configs/llm")
def create_llm_config(
        config: LLMConfigCreate,
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    new_config = LLMConfig(
        user_id=user_id,
        name=config.name,
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        folder_id=config.folder_id,
    )
    db.add(new_config)
    db.commit()
    db.refresh(new_config)
    return {'message': f"LLM Config Saved!", 'config_id': new_config.id}

@app.get("/configs/llm")
def get_llm_configs(
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    configs = db.query(LLMConfig).filter(LLMConfig.user_id == user_id).all()
    return configs

@app.get("/chats")
def get_chats(
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    chats = db.query(Chat).filter(Chat.user_id == user_id).all()
    return chats

@app.post("/chats")
def create_chat(
        chat_data: ChatCreate,
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    new_chat = Chat(id=str(uuid.uuid4()), user_id=user_id)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"chat_id": new_chat.id}

@app.get("/chats/{chat_id}")
def get_chat_history(
        chat_id: str,
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        return {'error': 'Нет чата'}
    return chat.messages

@app.post("/chats/{chat_id}/messages")
async def send_message(
        chat_id: str,
        payload: UserMessage,
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db),
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чата нет")

    llm_config = db.query(LLMConfig).filter(LLMConfig.user_id == user_id).first()
    if not llm_config:
        return {'error': 'нужен LLM Config'}

    history = [{'role': m.role, 'content': m.content} for m in chat.messages]
    final_text, updated_history = await run_agent(payload.message, llm_config, history, db, chat_id)

    db.query(Message).filter(Message.chat_id == chat_id).delete()
    for msg in updated_history:
        new_msg = Message(
            chat_id=chat_id,
            role=msg.get('role') if isinstance(msg, dict) else msg.role,
            content=msg.get('content') if isinstance(msg, dict) else msg.content,
        )
        db.add(new_msg)
    db.commit()
    return {'answer': final_text}

@app.post("/mcp-configs")
def create_mcp_config(
    config: MCPConfigCreate,
    user_id: str = Depends(verify_auth),
    db: Session = Depends(get_db)
):
    new_mcp = MCPConfig(
        user_id=user_id,
        name=config.name,
        url=config.url,
        token=config.token
    )
    db.add(new_mcp)
    db.commit()
    db.refresh(new_mcp)
    return {"message": "MCP Config сохранен", "mcp_id": new_mcp.id}

@app.get("/mcp-configs")
def get_mcp_configs(
    user_id: str = Depends(verify_auth),
    db: Session = Depends(get_db)
):
    configs = db.query(MCPConfig).filter(MCPConfig.user_id == user_id).all()
    return configs

@app.post("/chats/{chat_id}/mcp/{mcp_id}")
def connect_mcp_to_chat(
        chat_id: str,
        mcp_id: str,
        user_id: str = Depends(verify_auth),
        db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()

    try:
        mcp_id = int(mcp_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="неверный формат")
    mcp = db.query(MCPConfig).filter(MCPConfig.id == mcp_id, MCPConfig.user_id == user_id).first()

    if not chat or not mcp:
        raise HTTPException(status_code=404, detail="Chat or MCP not found")

    if mcp not in chat.mcp_servers:
        chat.mcp_servers.append(mcp)
        db.commit()

    return {"message": f"MCP {mcp.name} connected to chat {chat_id}"}

