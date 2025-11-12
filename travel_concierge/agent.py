# pylint: disable=logging-fstring-interpolation
import asyncio
import json
import os
import uuid

from typing import Any

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from travel_concierge.remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)
from travel_concierge.tools.memory import _load_precreated_itinerary


load_dotenv()


def convert_part(part: Part, tool_context: ToolContext):
    """Convert a part to text. Only text parts are supported."""
    if part.type == "text":
        return part.text

    return f"Unknown type: {part.type}"


def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to text."""
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
            "messageId": uuid.uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


class TravelHostAgent:
    """Host agent for coordinating travel planning across specialized A2A remote agents.

    This agent intelligently routes travel requests to remote agents based on:
    - Trip phase (pre-booking, pre-trip, in-trip, post-trip)
    - User intent (inspiration, planning, booking, preparation, support, feedback)
    - Available remote agents (discovered via A2A protocol)

    The agent maintains context about the user's travel profile and current itinerary,
    delegating specialized tasks to remote agents while transparently relaying responses.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self.agent = None

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)

                try:
                    card = await card_resolver.get_agent_card()
                    remote_conn = RemoteAgentConnections(card, address)
                    self.remote_agent_connections[card.name] = remote_conn
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No agents found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> "TravelHostAgent":
        """Create and asynchronously initialize an instance of the TravelHostAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Create an instance of the RoutingAgent."""
        return Agent(
            model="gemini-2.5-flash",
            name="root_agent",
            description="A Travel Conceirge using the services of multiple sub-agents",
            instruction=self.root_instruction,
            before_agent_callback=_load_precreated_itinerary,
            tools=[self.send_message],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate the root instruction for the RoutingAgent."""
        from datetime import datetime

        current_agent = self.check_active_agent(context)
        state = context.state

        # Extract context information
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get user profile context
        user_profile_context = state.get("user_profile", "No user profile available")
        if isinstance(user_profile_context, dict):
            user_profile_context = json.dumps(user_profile_context, indent=2)

        # Get trip information context
        itinerary = state.get("itinerary", {})
        if isinstance(itinerary, dict):
            trip_start = itinerary.get("start_date", "Not set")
            trip_end = itinerary.get("end_date", "Not set")
            trip_destination = itinerary.get("destination", "Not determined")
        else:
            trip_start = "Not set"
            trip_end = "Not set"
            trip_destination = "Not determined"

        trip_info_context = f"""
        Destination: {trip_destination}
        Start Date: {trip_start}
        End Date: {trip_end}
        Active Agent: {current_agent['active_agent']}
        """

        instruction = f"""
**Role:** You are an expert Travel Concierge. Your primary function is to help users plan and manage their travel, routing requests to specialized remote agents when available.

**Core Responsibilities:**

1. **Travel Planning:** Help users discover destinations, plan itineraries, find flights/hotels, and manage bookings.

2. **Trip Phase Detection:** Determine the current trip phase using the context below:
   - **Pre-Booking Phase** (no trip dates): Help with inspiration and planning
   - **Pre-Trip Phase** (before start_date): Assist with preparation tasks
   - **In-Trip Phase** (during the trip dates): Provide real-time support
   - **Post-Trip Phase** (after end_date): Collect feedback and preferences

3. **Context-Aware Assistance:** Use relevant contextual information (user preferences, trip details, conversation history) to provide personalized recommendations.

4. **Task Delegation:** When remote agents are available, delegate specialized tasks to appropriate agents via A2A protocol.

**Available Remote A2A Agents:**

{self.agents}

**Current Context:**
Current Time: {current_time}

User Profile:
{user_profile_context}

Trip Information:
{trip_info_context}

**Decision Logic:**

- For travel inspiration, destination suggestions, or activity recommendations → Use **inspiration_agent** if available
- For flight/hotel searches, seat/room selections, or itinerary building → Use **planning_agent** if available
- For reservations, payment processing, or booking confirmations → Use **booking_agent** if available
- For pre-trip preparation (visas, weather, packing) → Use **pre_trip_agent** if available
- For real-time travel support or daily itineraries → Use **in_trip_agent** if available
- For post-trip feedback or preference extraction → Use **post_trip_agent** if available

**Key Directives:**

✓ Provide helpful travel planning advice to users
✓ Route tasks to remote agents when available based on trip phase and user intent
✓ Provide minimal but complete context to remote agents
✓ Transparently relay all remote agent responses to the user
✓ Use the send_message tool to communicate with remote agents via A2A
✓ Handle long-running operations gracefully (agents may take time to respond)
✓ Focus on the most recent user interactions when making decisions
"""
        return instruction

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f'{state["active_agent"]}'}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to remote seller agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        task_id = state["task_id"] if "task_id" in state else str(uuid.uuid4())

        if "context_id" in state:
            context_id = state["context_id"]
        else:
            context_id = str(uuid.uuid4())

        message_id = ""
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])
            if "message_id" in state["input_message_metadata"]:
                message_id = state["input_message_metadata"]["message_id"]
        if not message_id:
            message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": task}
                ],  # Use the 'task' argument here
                "messageId": message_id,
            },
        }

        if task_id:
            payload["message"]["taskId"] = task_id

        if context_id:
            payload["message"]["contextId"] = context_id

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request=message_request
        )
        print(
            "send_response",
            send_response.model_dump_json(exclude_none=True, indent=2),
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return None

        if not isinstance(send_response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return None

        return send_response.root.result


def _get_initialized_routing_agent_sync() -> Agent:
    """Synchronously creates and initializes the RoutingAgent."""

    remote_agent_urls = [
        "http://0.0.0.0:10001",
        # "http://localhost:10003",  # Nate's Agent
        # "http://localhost:10004",
    ]

    async def _async_main() -> Agent:
        host_agent = await TravelHostAgent.create(remote_agent_urls)
        print(host_agent)
        return host_agent.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Event loop already running. Initializing agent without remote connections."
            )
            # # Create an agent without async initialization
            # instance = TravelHostAgent()
            # return instance.create_agent()
        else:
            raise


root_agent = _get_initialized_routing_agent_sync()
