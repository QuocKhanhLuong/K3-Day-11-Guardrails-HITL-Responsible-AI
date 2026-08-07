# OpenAI migration verification

Live runtime calls now use the OpenAI Responses API through `OPENAI_API_KEY`.

The remaining Google packages are retained only because the starter/public tests import
Google ADK plugin base classes and `types.Content`; no live Gemini call is made by:

- `src/main.py`
- `src/agents/agent.py`
- `src/agents/guards_agent.py`
- `src/guardrails/output_guardrails.py`
- `src/attacks/attacks.py`

Recommended environment:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```
