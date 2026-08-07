"""Reliable helpers for sending messages through Google ADK runners."""
from __future__ import annotations

from google.genai import types


def _content_text(content) -> str:
    """Extract text parts from an ADK/GenAI content object."""
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

    The stream is fully consumed so callbacks finish. Only the latest final model
    event is returned; tool/intermediate text is never concatenated into a fake
    answer. A requested session ID is preserved when a new session is created.
    """
    del agent  # Runner owns the active agent; retained for starter API compatibility.
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
