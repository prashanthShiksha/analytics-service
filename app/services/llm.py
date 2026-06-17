import json
import logging
import urllib.error
import urllib.request
from app.config import settings

logger = logging.getLogger("analytics_service.services.llm")

def openrouter_chat_completion(prompt: str) -> str:
    """
    Makes a synchronous HTTP request to OpenRouter to generate content.
    Independent of Temporal and can be called from CLI/scripts.
    """
    api_key = settings.OPENROUTER_API_KEY
    model = settings.OPENROUTER_MODEL

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    # Log the raw prompt to the terminal console for easy visibility
    logger.info(f"\n=================== [LLM CALL] RAW PROMPT SENT TO MODEL '{model}' ===================\n{prompt}\n=================================================================================")

    request_body = {
        "model": model,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/shikshalokam/analytics-temporal-poc",
        "X-Title": "analytics-temporal-poc",
    }

    request = urllib.request.Request(
        f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.LLM_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"OpenRouter HTTP Error {e.code}: {error_body}")
        raise RuntimeError(f"OpenRouter request failed with HTTP {e.code}: {error_body}") from e
    except Exception as e:
        logger.error(f"OpenRouter Connection Error: {e}")
        raise RuntimeError(f"Failed to connect to OpenRouter: {e}") from e

    try:
        choice = response_data["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response_data}") from e

    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) for part in content)

    if not content:
        raise RuntimeError("OpenRouter response did not include content.")

    return content.strip()
