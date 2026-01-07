"""
OpenRouter API Client for EASD Evaluation

Handles parallel calls to multiple judge models with retry logic and JSON repair.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

# OpenRouter model IDs
MODELS = {
    "gemini": "google/gemini-3-pro-preview",
    "grok": "x-ai/grok-4.1-fast",
    "llama": "meta-llama/llama-4-maverick",
}

# Stage 3 Synthesis model (separate from judge models)
SYNTHESIS_MODEL = {
    "opus": "anthropic/claude-opus-4.5",
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# JSON repair prompt - used when initial parse fails
JSON_REPAIR_PROMPT = """The following output was supposed to be valid JSON matching a specific schema, but it failed to parse.

Fix the JSON so it is valid. Do not change the meaning or values. Only fix structural JSON issues like:
- Missing quotes around strings
- Missing commas between elements
- Unclosed brackets or braces
- Invalid escape sequences
- Trailing commas

Return ONLY the fixed JSON, no explanation.

Broken output:
{broken_json}
"""


@dataclass
class JudgeResponse:
    """Response from a single judge model."""
    model_id: str
    model_alias: str
    success: bool
    evaluation: Optional[dict] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    usage: Optional[dict] = None
    repair_attempted: bool = False
    parse_attempts: int = 1


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        # Try loading from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY"):
                        key = line.split("=", 1)[1].strip()
                        break
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file")
    return key


def extract_json_from_response(text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Extract the FIRST valid JSON object from model response.
    
    Returns (parsed_dict, error_message).
    If successful, error_message is None.
    If failed, parsed_dict is None and error_message describes the issue.
    """
    if not text or not text.strip():
        return None, "Empty response"
    
    # Strategy 1: Try to find JSON in code blocks first (most common format)
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(code_block_pattern, text)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Try parsing the entire response as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError as e:
        pass
    
    # Strategy 3: Find the first { and try to extract a balanced JSON object
    first_brace = text.find('{')
    if first_brace != -1:
        # Count braces to find the matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[first_brace:], start=first_brace):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed, None
                    except json.JSONDecodeError:
                        break  # Found balanced braces but invalid JSON
        
    # Strategy 4: Last resort - find any JSON object pattern
    json_pattern = r"\{[\s\S]*\}"
    match = re.search(json_pattern, text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as e:
            return None, f"Found JSON-like structure but parse failed: {str(e)[:100]}"
    
    return None, "No valid JSON object found in response"


async def attempt_json_repair(
    client: httpx.AsyncClient,
    broken_json: str,
    api_key: str,
    timeout: float = 60.0
) -> tuple[Optional[dict], Optional[str]]:
    """
    Attempt to repair broken JSON using a model.
    
    Returns (parsed_dict, error_message).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://innovera.ai",
        "X-Title": "EASD Evaluator - JSON Repair"
    }
    
    # Use a fast, cheap model for repair (GPT-4.1-mini)
    repair_payload = {
        "model": "openai/gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": JSON_REPAIR_PROMPT.format(broken_json=broken_json[:8000])  # Limit size
            }
        ],
        "temperature": 0,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=repair_payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            repaired_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return extract_json_from_response(repaired_content)
        else:
            return None, f"Repair request failed: HTTP {response.status_code}"
            
    except Exception as e:
        return None, f"Repair request error: {str(e)}"


async def call_judge(
    client: httpx.AsyncClient,
    model_alias: str,
    model_id: str,
    prompt: str,
    api_key: str,
    max_retries: int = 2,
    timeout: float = 180.0,
    enable_repair: bool = True
) -> JudgeResponse:
    """
    Call a single judge model via OpenRouter with JSON repair fallback.
    
    Args:
        client: Async HTTP client
        model_alias: Short name (gemini, grok, gpt)
        model_id: Full OpenRouter model ID
        prompt: The evaluation prompt
        api_key: OpenRouter API key
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds
        enable_repair: Whether to attempt JSON repair on parse failure
    
    Returns:
        JudgeResponse with evaluation or error
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://innovera.ai",
        "X-Title": "EASD Evaluator"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,  # Low temperature for consistent scoring
        "max_tokens": 12000,
        "response_format": {"type": "json_object"}
    }
    
    last_error = None
    raw_content = None
    usage = None
    repair_attempted = False
    parse_attempts = 0
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                
                # Attempt 1: Direct parse
                parse_attempts = 1
                evaluation, parse_error = extract_json_from_response(raw_content)
                
                if evaluation:
                    return JudgeResponse(
                        model_id=model_id,
                        model_alias=model_alias,
                        success=True,
                        evaluation=evaluation,
                        raw_response=raw_content,
                        usage=usage,
                        repair_attempted=False,
                        parse_attempts=parse_attempts
                    )
                
                # Attempt 2: JSON repair (single retry only)
                if enable_repair and raw_content:
                    repair_attempted = True
                    parse_attempts = 2
                    repaired_evaluation, repair_error = await attempt_json_repair(
                        client, raw_content, api_key
                    )
                    
                    if repaired_evaluation:
                        return JudgeResponse(
                            model_id=model_id,
                            model_alias=model_alias,
                            success=True,
                            evaluation=repaired_evaluation,
                            raw_response=raw_content,
                            usage=usage,
                            repair_attempted=True,
                            parse_attempts=parse_attempts
                        )
                    
                    last_error = f"JSON parse failed: {parse_error}. Repair failed: {repair_error}"
                else:
                    last_error = f"JSON parse failed: {parse_error}"
                
                # Don't retry if we got a response but couldn't parse it
                # The model gave us something, it's just not valid JSON
                break
            
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                last_error = f"Rate limited (429), retrying..."
                continue
            
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except httpx.TimeoutException:
            last_error = f"Request timed out after {timeout}s"
        except Exception as e:
            last_error = str(e)
        
        # Wait before retry
        if attempt < max_retries:
            await asyncio.sleep(1)
    
    return JudgeResponse(
        model_id=model_id,
        model_alias=model_alias,
        success=False,
        error=last_error,
        raw_response=raw_content,
        usage=usage,
        repair_attempted=repair_attempted,
        parse_attempts=parse_attempts
    )


async def call_all_judges(
    prompt: str,
    models: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    enable_repair: bool = True
) -> list[JudgeResponse]:
    """
    Call all judge models in parallel.
    
    Args:
        prompt: The evaluation prompt
        models: List of model aliases to use (default: all)
        api_key: OpenRouter API key (default: from env)
        enable_repair: Whether to attempt JSON repair on parse failure
    
    Returns:
        List of JudgeResponse objects
    """
    if api_key is None:
        api_key = get_api_key()
    
    if models is None:
        models = list(MODELS.keys())
    
    async with httpx.AsyncClient() as client:
        tasks = [
            call_judge(
                client, alias, MODELS[alias], prompt, api_key,
                enable_repair=enable_repair
            )
            for alias in models
            if alias in MODELS
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                alias = models[i]
                results.append(JudgeResponse(
                    model_id=MODELS.get(alias, "unknown"),
                    model_alias=alias,
                    success=False,
                    error=str(resp)
                ))
            else:
                results.append(resp)
        
        return results


def call_judges_sync(
    prompt: str,
    models: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    enable_repair: bool = True
) -> list[JudgeResponse]:
    """
    Synchronous wrapper for call_all_judges.
    """
    return asyncio.run(call_all_judges(prompt, models, api_key, enable_repair))


async def call_synthesis_model(
    prompt: str,
    model_alias: str = "opus",
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    enable_repair: bool = True
) -> JudgeResponse:
    """
    Call the synthesis model (Claude Opus) for Stage 3 meta-analysis.
    
    Args:
        prompt: The synthesis prompt with all judge evaluations
        model_alias: Model alias (default: opus)
        api_key: OpenRouter API key (default: from env)
        timeout: Request timeout in seconds (longer for synthesis)
        enable_repair: Whether to attempt JSON repair on parse failure
    
    Returns:
        JudgeResponse with synthesis result or error
    """
    if api_key is None:
        api_key = get_api_key()
    
    model_id = SYNTHESIS_MODEL.get(model_alias)
    if not model_id:
        return JudgeResponse(
            model_id="unknown",
            model_alias=model_alias,
            success=False,
            error=f"Unknown synthesis model: {model_alias}"
        )
    
    async with httpx.AsyncClient() as client:
        return await call_judge(
            client,
            model_alias,
            model_id,
            prompt,
            api_key,
            max_retries=2,
            timeout=timeout,
            enable_repair=enable_repair
        )


def call_synthesis_sync(
    prompt: str,
    model_alias: str = "opus",
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    enable_repair: bool = True
) -> JudgeResponse:
    """
    Synchronous wrapper for call_synthesis_model.
    """
    return asyncio.run(call_synthesis_model(prompt, model_alias, api_key, timeout, enable_repair))
