from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Table
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

chat_mcp_association = Table(
    'chat_mcp',
    Base.metadata,
    Column('chat_id', String, ForeignKey('chats.id')),
    Column('mcp_id', Integer, ForeignKey('mcp_configs.id'))
)

class LLMConfig(Base):
    __tablename__ = 'llm_configs'
    __table_args__ = {
        'comment': 'LLM конфиги'
    }

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    name = Column(String)
    base_url = Column(String)
    api_key = Column(String)
    model = Column(String)
    folder_id = Column(String, nullable=True)

class MCPConfig(Base):
    __tablename__ = 'mcp_configs'
    __table_args__ = {
        'comment': 'MCP конфиги'
    }

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    name = Column(String)
    url = Column(String)
    token = Column(String)

class Chat(Base):
    __tablename__ = 'chats'
    __table_args__ = {
        'comment': 'чаты пользователей с темами'
    }

    id = Column(String, primary_key=True, comment='id чата')
    user_id = Column(String, index=True, comment='id юзера')
    #title = Column(String, comment='тема чата')
    created_at = Column(DateTime, default=datetime.now, comment='время создания чата')

    messages = relationship('Message', back_populates='chat', cascade='all, delete-orphan')
    mcp_servers = relationship("MCPConfig", secondary=chat_mcp_association)

class Message(Base):
    __tablename__ = 'messages'
    __table_args__ = {
        'comment': 'сообщения'
    }

    id = Column(Integer, primary_key=True, comment='идентификатор соо')
    chat_id = Column(String, ForeignKey("chats.id"), comment='какой чат')
    role = Column(String)
    content = Column(Text, comment='Суть соо')
    tool_call_id = Column(String, nullable=True, comment='для ответов инструментов')
    created_at = Column(DateTime, default=datetime.now, comment='время создания соо')

    chat = relationship("Chat", back_populates="messages")
