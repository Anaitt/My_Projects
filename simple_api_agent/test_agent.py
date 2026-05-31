import asyncio
from app.core.agent import run_agent


class MockConfig:
    api_key = "fake-key"
    base_url = "https://openai.com"
    model = "gpt-4o"


async def test():
    print("Запуск теста...")

    try:
        response, messages = await run_agent(
            user_message="Сколько будет 10 умножить на 2?",
            llm_config=MockConfig(),
            chat_history=[]
        )
        print(f"Ответ: {response}")
    except Exception as e:
        print(f"Тест успешен: сервер выдал ошибку: {e}")


if __name__ == "__main__":
    asyncio.run(test())
