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
# Batch evaluation helpers
# ---------------------------------------------------------

EVAL_DATASET_PATH = Path(
    os.getenv(
        "EVAL_DATASET_PATH",
        str(BASE_DIR / "data" / "agent-optimizer-eval.jsonl"),
    )
)

RESULTS_PATH = Path(
    os.getenv(
        "EVAL_RESULTS_PATH",
        str(BASE_DIR / "agent-optimizer-baseline-results.jsonl"),
    )
)

MAX_RUNS = int(os.getenv("EVAL_MAX_RUNS", "40"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(
                    f"Line {line_number} must contain 'id' and 'prompt'."
                )
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Run one isolated single-turn evaluation
# ---------------------------------------------------------

def run_single_turn(user_message: str) -> tuple[str, list[str], str]:
    # Fresh conversation for every row = no cross-example context.
    conversation = openai_client.conversations.create()
    tools_called: list[str] = []

    response = openai_client.responses.create(
        conversation=conversation.id,
        input=[{"role": "user", "content": user_message}],
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "version": agent.version,
                "type": "agent_reference",
            }
        },
    )

    while True:
        function_outputs: ResponseInputParam = []

        for item in response.output:
            if item.type == "function_call":
                tools_called.append(item.name)
                print(f"    [tool] {item.name}({item.arguments})")
                function_outputs.append(execute_function_call(item))

        # Hosted File Search runs inside Foundry and does not need
        # local dispatch. If there are no local function calls left,
        # the model has reached its final answer.
        if not function_outputs:
            return response.output_text, tools_called, conversation.id

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
# Batch runner
# ---------------------------------------------------------

def main():
    dataset = load_jsonl(EVAL_DATASET_PATH)[:MAX_RUNS]

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print()
    print("Risk Advisor Baseline Evaluation")
    print(f"Dataset: {EVAL_DATASET_PATH}")
    print(f"Runs: {len(dataset)}")
    print(f"Results: {RESULTS_PATH}")
    print()

    succeeded = 0
    failed = 0

    for index, item in enumerate(dataset, start=1):
        item_id = item["id"]
        prompt = item["prompt"]
        result_row = dict(item)
        trace_id = None

        print(f"[{index}/{len(dataset)}] {item_id}")
        print(f"    Prompt: {prompt}")

        try:
            # Each row gets its own root application span and therefore
            # its own trace ID.
            with tracer.start_as_current_span(
                f"evaluation.{item_id}"
            ) as span:
                span.set_attribute("evaluation.id", item_id)
                span.set_attribute("evaluation.prompt", prompt)

                trace_id = format(
                    span.get_span_context().trace_id,
                    "032x",
                )

                answer, tools_called, conversation_id = run_single_turn(prompt)

                span.set_attribute("evaluation.success", True)
                span.set_attribute(
                    "gen_ai.conversation.id",
                    conversation_id,
                )

            result_row.update(
                {
                    "agent_name": agent.name,
                    "agent_version": agent.version,
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "response": answer,
                    "local_function_tools_called": tools_called,
                    "status": "success",
                }
            )
            succeeded += 1

            print(f"    Trace ID: {trace_id}")
            print(
                "    Local tools: "
                + (", ".join(tools_called) if tools_called else "none")
            )
            print()

        except Exception as exc:
            failed += 1
            result_row.update(
                {
                    "agent_name": agent.name,
                    "agent_version": agent.version,
                    "trace_id": trace_id,
                    "response": None,
                    "local_function_tools_called": [],
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"    ERROR: {exc}\n")

        # Write each result immediately so partial progress is preserved.
        append_jsonl(RESULTS_PATH, result_row)

    # Flush telemetry before this short-lived batch process exits.
    tracer_provider = trace.get_tracer_provider()
    force_flush = getattr(tracer_provider, "force_flush", None)
    if callable(force_flush):
        force_flush()

    print("----------------------------------------")
    print("Evaluation complete")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Results:   {RESULTS_PATH}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()