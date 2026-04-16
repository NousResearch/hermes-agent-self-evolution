"""DSPy LM provider that delegates to Codex CLI (GPT-5.4 via ChatGPT auth)."""

import json
import subprocess

import dspy


class CodexLM(dspy.BaseLM):
    """Wraps `codex exec --json` as a DSPy language model.

    Uses the local codex CLI which authenticates via ChatGPT Plus OAuth,
    giving access to GPT-5.4 without needing an API key.
    """

    def __init__(self, timeout: int = 300, **kwargs):
        self.timeout = timeout
        self.model = "codex/gpt-5.4"
        self.model_type = "chat"
        self.cache = kwargs.get("cache", True)
        self.history = []
        self.kwargs = kwargs

    def __call__(self, prompt=None, messages=None, **kwargs) -> list[str]:
        if messages:
            # Convert chat messages to a single prompt string
            text = "\n\n".join(f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in messages)
        elif prompt:
            text = prompt if isinstance(prompt, str) else str(prompt)
        else:
            return [""]

        try:
            result = subprocess.run(
                ["codex", "exec", "--json", text],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                stdin=subprocess.DEVNULL,
            )

            if result.returncode != 0:
                return [f"Error: codex exec failed: {result.stderr[:200]}"]

            # Parse NDJSON output, find the agent_message
            response_text = ""
            for line in result.stdout.strip().split("\n"):
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "item.completed":
                        item = obj.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                except json.JSONDecodeError:
                    continue

            self.history.append({"prompt": text[:200], "response": response_text[:200]})
            return [response_text]

        except subprocess.TimeoutExpired:
            return ["Error: codex exec timed out"]
        except Exception as e:
            return [f"Error: {e}"]

    def inspect_history(self, n: int = 1):
        return self.history[-n:]

    @property
    def provider(self):
        return "codex"
