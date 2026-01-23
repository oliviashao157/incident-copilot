"""LLM client abstraction for OpenAI and Anthropic."""

from typing import Optional

from src.config import LLMProvider, get_settings


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider to use (from settings if not specified)
            model: Model name (from settings if not specified)
        """
        self.settings = get_settings()
        self.provider = provider or self.settings.llm_provider
        self.model = model or self.settings.get_llm_model()
        self._client = None

    def _get_client(self):
        """Lazily initialize the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == LLMProvider.ANTHROPIC:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        else:
            import openai

            self._client = openai.OpenAI(api_key=self.settings.openai_api_key)

        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        client = self._get_client()

        if self.provider == LLMProvider.ANTHROPIC:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text

        else:  # OpenAI
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate a JSON response from the LLM.

        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON dict
        """
        import json

        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nRespond only with valid JSON, no other text."

        response = self.generate(
            prompt=prompt,
            system_prompt=json_system.strip(),
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for structured output
        )

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            # Find JSON object
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            return json.loads(response)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")
