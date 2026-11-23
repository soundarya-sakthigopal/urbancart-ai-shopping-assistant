from openai import OpenAI
from src.config.settings import OPENAI_API_KEY


def test_openai_connection():

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-4o-mini",
        input="Reply with exactly: UrbanCart OpenAI connection successful."
    )

    return response.output_text