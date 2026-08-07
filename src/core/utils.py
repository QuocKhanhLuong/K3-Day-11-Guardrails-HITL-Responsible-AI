"""Reliable helpers for OpenAI runners and legacy ADK-compatible tests."""
from __future__ import annotations

from google.genai import types


def _content_text(content) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    return "".join(
        part.text for part in content.parts if getattr(part, "text", None)
    )


async def chat_with_agent(
    agent,
    runner,
    user_message: str,
    session_id: str | None = None,
    *,
    user_id: str = "student",
):
    """Send one message and return ``(response_text, session)``.

    New runtime adapters expose ``runner.chat``. The ADK branch remains only for
    starter/public-test compatibility and can still consume final ADK events.
    """
    if callable(getattr(runner, "chat", None)):
        return await runner.chat(
            str(user_message), session_id=session_id, user_id=user_id
        )

    del agent
    app_name = runner.app_name
    session = None

    if session_id is not None:
        session = await runner.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    if session is None:
        create_kwargs = {"app_name": app_name, "user_id": user_id}
        if session_id is not None:
            create_kwargs["session_id"] = session_id
        session = await runner.session_service.create_session(**create_kwargs)

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=str(user_message))],
    )

    final_text = ""
    last_model_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        error_code = getattr(event, "error_code", None)
        error_message = getattr(event, "error_message", None)
        if error_code or error_message:
            raise RuntimeError(
                f"ADK event failed: {error_code or 'unknown'}: "
                f"{error_message or 'no message'}"
            )

        event_content = getattr(event, "content", None)
        text = _content_text(event_content)
        if not text:
            continue
        if getattr(event_content, "role", None) == "model":
            last_model_text = text
        is_final = getattr(event, "is_final_response", None)
        if callable(is_final) and is_final():
            final_text = text

    response = final_text or last_model_text
    if not response:
        raise RuntimeError("Agent produced no final text response")
    return response, session
