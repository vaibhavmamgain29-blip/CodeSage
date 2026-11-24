import os
import json
import time
import re
from typing import Optional, Dict, Any, List
from codesage.config import settings

class LLMResponse:
    def __init__(self, content: str, prompt_tokens: int = 0, completion_tokens: int = 0, latency_sec: float = 0.0):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_sec = latency_sec

    def extract_json(self) -> Dict[str, Any]:
        """Robustly extracts JSON object or array from the LLM output text."""
        text = self.content.strip()

        # Handle ```json markdown fences
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

        # Direct json load
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the outermost { ... } or [ ... ]
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                try:
                    return json.loads(text[start_obj:end_obj + 1])
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Failed to parse valid JSON from LLM response:\n{self.content}")


class LLMClient:
    """Unified LLM client with retries, timeout, and usage tracking."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.api_key = api_key or self._get_api_key_for_provider()

    def _get_api_key_for_provider(self) -> str:
        if self.provider == "gemini":
            return settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        elif self.provider == "openai":
            return settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        return ""

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        retries: int = 2
    ) -> LLMResponse:
        """Send generation request with automatic retries on transient errors."""
        last_error = None

        for attempt in range(retries + 1):
            start_time = time.time()
            try:
                if self.provider == "gemini":
                    return self._call_gemini(prompt, system_instruction, temperature, max_tokens, start_time)
                elif self.provider == "openai":
                    return self._call_openai(prompt, system_instruction, temperature, max_tokens, start_time)
                elif self.provider == "ollama":
                    return self._call_ollama(prompt, system_instruction, temperature, max_tokens, start_time)
                elif self.provider == "mock":
                    return self._call_mock(prompt, system_instruction, start_time)
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.provider}")
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            latency = time.time() - start_time
            # Token usage from response metadata if available
            p_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) if hasattr(resp, "usage_metadata") else 0
            c_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) if hasattr(resp, "usage_metadata") else 0
            return LLMResponse(content=resp.text or "", prompt_tokens=p_tokens, completion_tokens=c_tokens, latency_sec=latency)
        except Exception:
            # Fallback to direct REST API if google-genai is not configured or fails
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            res = requests.post(url, json=payload, timeout=45)
            res.raise_for_status()
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            latency = time.time() - start_time
            return LLMResponse(content=text, latency_sec=latency)

    def _call_openai(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        latency = time.time() - start_time
        usage = resp.usage
        p_tokens = usage.prompt_tokens if usage else 0
        c_tokens = usage.completion_tokens if usage else 0
        text = resp.choices[0].message.content or ""
        return LLMResponse(content=text, prompt_tokens=p_tokens, completion_tokens=c_tokens, latency_sec=latency)

    def _call_ollama(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        import requests
        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        return LLMResponse(
            content=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_sec=latency
        )

    def _call_mock(
        self,
        prompt: str,
        system_instruction: Optional[str],
        start_time: float
    ) -> LLMResponse:
        """Deterministic mock for tests and offline evaluation."""
        latency = time.time() - start_time
        if "REVIEW" in (system_instruction or "") or "code review" in (system_instruction or "").lower():
            findings = []
            prompt_lower = prompt.lower()

            if "select" in prompt_lower and "email" in prompt_lower:
                findings.append({
                    "file": "app/db/users.py",
                    "line": 15,
                    "category": "security",
                    "severity": "high",
                    "confidence": 0.98,
                    "issue": "SQL injection vulnerability caused by raw string formatting in SQL query.",
                    "suggestion": "Use parameterized queries: cursor.execute('SELECT ... WHERE email = %s', (email,))"
                })
            elif "binary_search" in prompt_lower or "high = len(arr)" in prompt_lower:
                findings.append({
                    "file": "app/utils/search.py",
                    "line": 10,
                    "category": "bug",
                    "severity": "high",
                    "confidence": 0.95,
                    "issue": "Off-by-one boundary error: high should be initialized to len(arr) - 1, otherwise IndexError can occur.",
                    "suggestion": "Set high = len(arr) - 1"
                })
            elif "user_repo.find_by_id" in prompt_lower or "profile.py" in prompt_lower:
                findings.append({
                    "file": "app/services/profile.py",
                    "line": 21,
                    "category": "bug",
                    "severity": "medium",
                    "confidence": 0.92,
                    "issue": "Missing None check for user before accessing first_name and last_name.",
                    "suggestion": "Add if not user: raise UserNotFoundError() before accessing attributes."
                })
            elif "aws_secret_access_key" in prompt_lower or "boto3" in prompt_lower:
                findings.append({
                    "file": "app/integrations/aws_client.py",
                    "line": 6,
                    "category": "security",
                    "severity": "high",
                    "confidence": 0.99,
                    "issue": "Hardcoded AWS access key and secret key in source code.",
                    "suggestion": "Load credentials from environment variables or IAM role."
                })
            elif "download_file" in prompt_lower or "/var/data/uploads" in prompt_lower:
                findings.append({
                    "file": "app/controllers/download.py",
                    "line": 12,
                    "category": "security",
                    "severity": "high",
                    "confidence": 0.96,
                    "issue": "Path traversal vulnerability: filename is directly concatenated into filepath without sanitization.",
                    "suggestion": "Use os.path.basename(filename) or validate path with os.path.commonpath."
                })
            else:
                findings.append({
                    "file": "sample.py",
                    "line": 10,
                    "category": "bug",
                    "severity": "medium",
                    "confidence": 0.88,
                    "issue": "Potential NoneType dereference if item is missing.",
                    "suggestion": "Check if item is not None before accessing attributes."
                })

            mock_data = {
                "summary": f"Completed review of diff with {len(findings)} finding(s).",
                "findings": findings
            }
            return LLMResponse(content=json.dumps(mock_data), prompt_tokens=150, completion_tokens=80, latency_sec=latency)
        else:
            # Q&A answer mock
            ans = "Based on the repository code, this is handled in `sample.py:10-25`.\n\nCitation: sample.py:10-25"
            return LLMResponse(content=ans, prompt_tokens=60, completion_tokens=30, latency_sec=latency)
