import os
from fastapi import Header, HTTPException, status

STATIC_AUTH_TOKEN = os.getenv("AUTH_TOKEN")
#RADISH

async def verify_auth(
        x_api_key: str = Header(None, alias='X-API-Key'),
        x_user_id: str = Header(None, alias='X-User-Id'),
):
    if not STATIC_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="А где токен?",
        )

    if not x_api_key or not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Нужно ввести X-API-Key или X-User-Id :(',
        )

    if x_api_key != STATIC_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Неправильный API Key :('
        )

    return x_user_id