"""LLM API client with support for OpenAI and OpenRouter."""

import os
import time
import openai
import httpx

# OpenAI client
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# OpenRouter client (for Claude, etc.)
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_KEY"),
)


def call_llm(messages, model="gpt-4.1", temperature=0.0, max_tokens=2048, provider="openai"):
    """Call an LLM with retry logic. Returns the response text."""
    client = openai_client if provider == "openai" else openrouter_client

    for attempt in range(8):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            wait = min(5 * (2 ** attempt), 120)
            print(f"  Rate limit/timeout (attempt {attempt+1}), waiting {wait}s...")
            time.sleep(wait)
        except openai.APIError as e:
            wait = min(5 * (2 ** attempt), 120)
            print(f"  API error (attempt {attempt+1}), waiting {wait}s... ({e})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after 8 retries for model {model}")
