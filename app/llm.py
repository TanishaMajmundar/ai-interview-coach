import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL")


def call_llm(system: str, user: str, json_mode: bool = False,
             max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Send one system + user message and return the reply text."""
    kwargs = {}
    if json_mode:
        # Forces valid JSON syntax (the prompt must also mention "JSON")
        kwargs["response_format"] = {"type": "json_object"}

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content or ""