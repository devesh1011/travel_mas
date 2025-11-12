import asyncio
import traceback  # Import the traceback module

from collections.abc import AsyncIterator
from pprint import pformat

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from travel_concierge.agent import (
    root_agent as travel_host_agent,
)


APP_NAME = "routing_app"
USER_ID = "default_user"
SESSION_ID = "default_session"

SESSION_SERVICE = InMemorySessionService()
ROUTING_AGENT_RUNNER = Runner(
    agent=travel_host_agent,
    app_name=APP_NAME,
    session_service=SESSION_SERVICE,
)


async def get_response_from_agent(
    message: str,
    history: list,
) -> AsyncIterator[dict]:
    """Get response from host agent."""
    try:
        event_iterator: AsyncIterator[Event] = ROUTING_AGENT_RUNNER.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        )

        async for event in event_iterator:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        formatted_call = f"```python\n{pformat(part.function_call.model_dump(exclude_none=True), indent=2, width=80)}\n```"
                        yield {
                            "role": "assistant",
                            "content": f"🛠️ **Tool Call: {part.function_call.name}**\n{formatted_call}",
                        }
                    elif part.function_response:
                        response_content = part.function_response.response
                        if (
                            isinstance(response_content, dict)
                            and "response" in response_content
                        ):
                            formatted_response_data = response_content["response"]
                        else:
                            formatted_response_data = response_content
                        formatted_response = f"```json\n{pformat(formatted_response_data, indent=2, width=80)}\n```"
                        yield {
                            "role": "assistant",
                            "content": f"⚡ **Tool Response from {part.function_response.name}**\n{formatted_response}",
                        }
            if event.is_final_response():
                final_response_text = ""
                if event.content and event.content.parts:
                    final_response_text = "".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                elif event.actions and event.actions.escalate:
                    final_response_text = f'Agent escalated: {event.error_message or "No specific message."}'
                if final_response_text:
                    yield {"role": "assistant", "content": final_response_text}
                break
    except Exception as e:
        print(f"Error in get_response_from_agent (Type: {type(e)}): {e}")
        traceback.print_exc()  # This will print the full traceback
        yield {
            "role": "assistant",
            "content": "An error occurred while processing your request. Please check the server logs for details.",
        }
