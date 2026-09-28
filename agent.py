# pip install \
#   "azure-ai-projects>=2.3.0" \
#   azure-identity \
#   azure-monitor-opentelemetry \
#   opentelemetry-api \
#   python-dotenv

import json
import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    FileSearchTool,
    FunctionTool,
    PromptAgentDefinition,
)
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)
from opentelemetry import trace


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.getenv(
    "FOUNDRY_MODEL_DEPLOYMENT",
    "gpt-5.6-sol-1",
)

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

AGENT_NAME = "risk-advisor-agent"

INSTRUCTIONS_PATH = BASE_DIR / "agent-instructions.md"

DATA_DIR = BASE_DIR / "data"
CLAIMS_PATH = DATA_DIR / "claims.json"
INSURANCE_PATH = DATA_DIR / "insurance-programs.json"
FACILITIES_PATH = DATA_DIR / "facility-exposures.json"
SUPPLY_CHAIN_PATH = DATA_DIR / "supply-chain-exposures.json"


# ---------------------------------------------------------
# Load agent instructions
# ---------------------------------------------------------

AGENT_INSTRUCTIONS = INSTRUCTIONS_PATH.read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------
# Load local structured data
# ---------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


CLAIMS_DATA = load_json(CLAIMS_PATH)
INSURANCE_DATA = load_json(INSURANCE_PATH)
FACILITY_DATA = load_json(FACILITIES_PATH)
SUPPLY_CHAIN_DATA = load_json(SUPPLY_CHAIN_PATH)


# ---------------------------------------------------------
# Local function implementations
# ---------------------------------------------------------

def get_claim_history(client_name: str) -> dict:
    """Return structured claims and historical loss data."""

    return {
        "client_name": client_name,
        "data": CLAIMS_DATA.get(client_name),
    }


def get_insurance_program(client_name: str) -> dict:
    """Return structured insurance program information."""

    return {
        "client_name": client_name,
        "data": INSURANCE_DATA.get(client_name),
    }


def get_facility_exposure(client_name: str) -> dict:
    """Return structured facility and operational exposure data."""

    return {
        "client_name": client_name,
        "data": FACILITY_DATA.get(client_name),
    }


def get_supply_chain_exposure(client_name: str) -> dict:
    """Return structured supply-chain exposure data."""

    return {
        "client_name": client_name,
        "data": SUPPLY_CHAIN_DATA.get(client_name),
    }


# Map model-requested function names to Python implementations.
FUNCTION_DISPATCH = {
    "get_claim_history": get_claim_history,
    "get_insurance_program": get_insurance_program,
    "get_facility_exposure": get_facility_exposure,
    "get_supply_chain_exposure": get_supply_chain_exposure,
}


# ---------------------------------------------------------
# Function tool definitions
#
# These descriptions intentionally remain fairly simple.
# Agent Optimizer can later improve these descriptions.
# ---------------------------------------------------------

get_claim_history_tool = FunctionTool(
    name="get_claim_history",
    description=(
        "Get historical claims and loss events for a client."
    ),
    parameters={
        "type": "object",
        "properties": {
            "client_name": {
                "type": "string",
                "description": (
                    "The name of the client whose claims "
                    "history should be retrieved."
                ),
            }
        },
        "required": ["client_name"],
        "additionalProperties": False,
    },
    strict=True,
)


get_insurance_program_tool = FunctionTool(
    name="get_insurance_program",
    description=(
        "Get the client's current insurance program."
    ),
    parameters={
        "type": "object",
        "properties": {
            "client_name": {
                "type": "string",
                "description": (
                    "The name of the client whose insurance "
                    "program should be retrieved."
                ),
            }
        },
        "required": ["client_name"],
        "additionalProperties": False,
    },
    strict=True,
)


get_facility_exposure_tool = FunctionTool(
    name="get_facility_exposure",
    description=(
        "Get information about the client's physical "
        "facilities and operational exposures."
    ),
    parameters={
        "type": "object",
        "properties": {
            "client_name": {
                "type": "string",
                "description": (
                    "The name of the client whose facility "
                    "exposure information should be retrieved."
                ),
            }
        },
        "required": ["client_name"],
        "additionalProperties": False,
    },
    strict=True,
)


get_supply_chain_exposure_tool = FunctionTool(
    name="get_supply_chain_exposure",
    description=(
        "Get information about the client's supply-chain "
        "dependencies and exposures."
    ),
    parameters={
        "type": "object",
        "properties": {
            "client_name": {
                "type": "string",
                "description": (
                    "The name of the client whose supply-chain "
                    "exposure should be retrieved."
                ),
            }
        },
        "required": ["client_name"],
        "additionalProperties": False,
    },
    strict=True,
)


# ---------------------------------------------------------
# Foundry client + tracing
# ---------------------------------------------------------

credential = DefaultAzureCredential()

project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
)


# Foundry server-side tracing will already capture the Prompt
# Agent run once Application Insights is connected to the project.
#
# This additional client-side tracing lets us create spans around
# the Python functions that execute locally.

connection_string = (
    project_client.telemetry
    .get_application_insights_connection_string()
)

if connection_string:
    configure_azure_monitor(
        connection_string=connection_string
    )

tracer = trace.get_tracer("risk-advisor-demo")

openai_client = project_client.get_openai_client()


# ---------------------------------------------------------
# Create a new Prompt Agent version
# ---------------------------------------------------------

agent = project_client.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model=MODEL_DEPLOYMENT,
        instructions=AGENT_INSTRUCTIONS,
        tools=[
            # Unstructured company profiles
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID]
            ),

            # Structured local data
            get_claim_history_tool,
            get_insurance_program_tool,
            get_facility_exposure_tool,
            get_supply_chain_exposure_tool,
        ],
    ),
    description=(
        "Risk Advisor Copilot used for Agent Optimizer demo."
    ),
)

print(
    f"Using agent: {agent.name}, "
    f"version: {agent.version}"
)


# ---------------------------------------------------------
# Create persistent conversation
# ---------------------------------------------------------

conversation = openai_client.conversations.create()

print(f"Conversation ID: {conversation.id}")


# ---------------------------------------------------------
# Execute a local function requested by the agent
# ---------------------------------------------------------

def execute_function_call(item) -> FunctionCallOutput:

    function_name = item.name

    if function_name not in FUNCTION_DISPATCH:
        raise ValueError(
            f"Unknown function requested: {function_name}"
        )

    arguments = json.loads(item.arguments)

    function = FUNCTION_DISPATCH[function_name]

    # This span covers the LOCAL execution of the tool.
    #
    # The Foundry server trace will show that the model requested
    # a function call. This custom span gives you visibility into
    # the actual Python lookup as well.
    with tracer.start_as_current_span(
        f"tool.{function_name}"
    ) as span:

        span.set_attribute(
            "gen_ai.tool.name",
            function_name,
        )

        span.set_attribute(
            "client_name",
            arguments.get("client_name", ""),
        )

        result = function(**arguments)

        span.set_attribute(
            "tool.result.found",
            result.get("data") is not None,
        )

    return FunctionCallOutput(
        type="function_call_output",
        call_id=item.call_id,
        output=json.dumps(result),
    )


# ---------------------------------------------------------
# Send one conversational turn
# ---------------------------------------------------------

def run_turn(user_message: str) -> str:

    response = openai_client.responses.create(
        conversation=conversation.id,
        input=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "version": agent.version,
                "type": "agent_reference",
            }
        },
    )

    # The agent can request one or more functions.
    # Keep resolving calls until the model produces its
    # final natural-language response.

    while True:

        function_outputs: ResponseInputParam = []

        for item in response.output:

            if item.type == "function_call":

                print(
                    f"[tool call] {item.name}"
                    f"({item.arguments})"
                )

                function_outputs.append(
                    execute_function_call(item)
                )

        # No local function calls means we have a final response.
        #
        # File Search is hosted by Foundry, so it executes
        # automatically and does not require handling here.
        if not function_outputs:
            return response.output_text

        # Send local tool results back into the SAME conversation.
        response = openai_client.responses.create(
            conversation=conversation.id,
            input=function_outputs,
            extra_body={
                "agent_reference": {
                    "name": agent.name,
                    "version": agent.version,
                    "type": "agent_reference",
                }
            },
        )


# ---------------------------------------------------------
# Interactive terminal chat
# ---------------------------------------------------------

def main():

    print()
    print("Risk Advisor Copilot")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            break

        if not user_input:
            continue

        try:

            with tracer.start_as_current_span(
                "risk-advisor-conversation-turn"
            ):

                answer = run_turn(user_input)

            print(f"\nAgent: {answer}\n")

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()