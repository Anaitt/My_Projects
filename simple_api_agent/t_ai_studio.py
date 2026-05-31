import openai

YANDEX_FOLDER_ID='b1gk2pnbt9eg066t4qfj'
YANDEX_API_KEY='AQVNwumVYbj_r0ae_ISxI0bRVQIcCiz5OWNA6T5F'

client = openai.OpenAI(
    api_key=YANDEX_API_KEY,
    project=YANDEX_FOLDER_ID,
    base_url="https://ai.api.cloud.yandex.net/v1"
)

YANDEX_MODEL = "yandexgpt-lite/latest"

response = client.responses.create(
    model=f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
    input="Придумай 3 необычные идеи для стартапа в сфере путешествий.",
    temperature=0.8,
    max_output_tokens=1500
)

print(response.output[0].content[0].text)