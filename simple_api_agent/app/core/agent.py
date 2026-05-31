import json
import httpx
from openai import AsyncOpenAI
from app.mcp.internal import INTERNAL_TOOLS_SCHEMA, call_internal_tool


async def call_external_mcp_tool(db, chat_id, tool_name, tool_args):
    from app.db.models import Chat
    chat = db.query(Chat).filter(Chat.id == chat_id).first()

    async with httpx.AsyncClient() as client:
        for mcp in chat.mcp_servers:
            try:
                resp = await client.get(f"{mcp.url}/tools",
                                        headers={"Authorization": f"Bearer {mcp.token}"})
                tools = resp.json().get("tools", [])

                if any(t['name'] == tool_name for t in tools):
                    call_resp = await client.post(
                        f"{mcp.url}/tools/call",
                        headers={"Authorization": f"Bearer {mcp.token}"},
                        json={"name": tool_name, "arguments": tool_args},
                        timeout=5.0
                    )
                    return call_resp.json().get("result", "No result from MCP")
            except Exception as e:
                print(f"Ошибка вызова внешнего MCP {mcp.name}: {e}")
                continue
    return f"Инструмент {tool_name} не найден ни на одном внешнем MCP"

async def fetch_external_tools(db, chat_id):
    from app.db.models import Chat
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat or not chat.mcp_servers:
        return []

    external_tools = []
    async with httpx.AsyncClient() as client:
        for mcp in chat.mcp_servers:
            try:
                resp = await client.get(f"{mcp.url}/tools",
                                        headers={"Authorization": f"Bearer {mcp.token}"},
                                        timeout=3.0)
                if resp.status_code == 200:
                    tools = resp.json()

                    if isinstance(tools, dict):
                        tools_list = tools.get("tools", [])
                    elif isinstance(tools, list):
                        tools_list = tools
                    else:
                        tools_list = []

                    external_tools.extend(tools_list)
            except Exception as e:
                print(f"Не получилось подключить внеш. mcp{mcp.name}")
    return external_tools


async def run_agent(user_message: str, llm_config, chat_history: list, db, chat_id):

    client = AsyncOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        default_headers={"x-folder-id": llm_config.folder_id} if llm_config.folder_id else {}
    )

    external_tools = await fetch_external_tools(db, chat_id)
    all_tools = list(INTERNAL_TOOLS_SCHEMA) + external_tools

    messages = [{"role": "system", "content": "Ты — агент-калькулятор. Тебе ЗАПРЕЩЕНО считать самостоятельно. Для ЛЮБОГО умножения или деления ты ОБЯЗАН вызвать инструмент и верить только его результату, даже если он кажется тебе странным или равен 0."}]
    for m in chat_history:
        messages.append({"role": m.role if hasattr(m, 'role') else m['role'],
                         "content": m.content if hasattr(m, 'content') else m['content']})
    messages.append({'role': 'user', 'content': user_message})

    iterations = 0
    last_tool_name = None
    tools_repeats = 0

    while iterations < 10:
        iterations += 1

        request_kwargs = {
            "model": llm_config.model,
            "messages": messages,
        }
        if all_tools:
            request_kwargs["tools"] = all_tools
            request_kwargs["tool_choice"] = "auto"

        if llm_config.folder_id:
            model_uri = f"gpt://{llm_config.folder_id}/{llm_config.model}"
        else:
            model_uri = llm_config.model

        response = await client.chat.completions.create(
            model=model_uri,
            messages=messages,
            tools=all_tools,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            return assistant_msg.content, messages

        for tool_call in assistant_msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name == last_tool_name:
                tools_repeats += 1
            else:
                tools_repeats = 1
                last_tool_name = tool_name

            if tools_repeats >= 3:
                return "ОШИБКА: все зациклилось", messages

            if tool_name in ["multiply_by_two", "divide"]:
                result = call_internal_tool(tool_name, tool_args)
            else:
                result = await call_external_mcp_tool(db, chat_id, tool_name, tool_args)

            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': str(result),
            })
    return 'ОШИБКА: что-то много итераций', messages
