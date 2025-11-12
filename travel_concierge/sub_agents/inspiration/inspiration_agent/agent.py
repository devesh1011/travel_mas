# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inspiration agent. A pre-booking agent covering the ideation part of the trip."""

import os
import sys
from pathlib import Path
from starlette.routing import Route
from starlette.responses import JSONResponse

# Add current directory to path to allow imports as a standalone package
sys.path.insert(0, str(Path(__file__).parent))

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from models import (
    DestinationIdeas,
    POISuggestions,
    json_response_config,
)
import prompt
from tools import map_tool


place_agent = Agent(
    model="gemini-2.5-flash",
    name="place_agent",
    instruction=prompt.PLACE_AGENT_INSTR,
    description="This agent suggests a few destination given some user preferences",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=DestinationIdeas,
    output_key="place",
    generate_content_config=json_response_config,
)

poi_agent = Agent(
    model="gemini-2.5-flash",
    name="poi_agent",
    description="This agent suggests a few activities and points of interests given a destination",
    instruction=prompt.POI_AGENT_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=POISuggestions,
    output_key="poi",
    generate_content_config=json_response_config,
)


def create_inspiration_agent() -> Agent:
    """Constructs the ADK agent."""

    return Agent(
        model="gemini-2.5-flash",
        name="inspiration_agent",
        description="A travel inspiration agent who inspire users, and discover their next vacations; Provide information about places, activities, interests,",
        instruction=prompt.INSPIRATION_AGENT_INSTR,
        tools=[AgentTool(agent=place_agent), AgentTool(agent=poi_agent), map_tool],
    )


root_agent = create_inspiration_agent()
