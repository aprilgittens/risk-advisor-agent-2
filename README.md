# Risk Advisor

Risk Advisor is a Python proof of concept that uses a Microsoft Foundry prompt
agent to analyze client risk information. It combines:

- hosted file search over company profile documents in a Foundry vector store;
- local function tools backed by structured JSON data;
- an interactive command-line experience;
- a batch evaluation runner for collecting responses, tool usage, conversation
  IDs, and trace IDs; and
- OpenTelemetry tracing through the Application Insights connection associated
  with the Foundry project.

## Repository contents

| Path | Purpose |
| --- | --- |
| `agent.py` | Starts an interactive terminal conversation with the risk advisor. |
| `agent-v2.py` | Runs isolated prompts from a JSONL evaluation dataset and writes JSONL results. |
| `agent-instructions.md` | System instructions used when each prompt-agent version is created. |
| `data/*.json` | Structured claims, insurance program, facility, and supply-chain data exposed through local function tools. |
| `data/agent-optimizer-eval.jsonl` | Default evaluation input used by `agent-v2.py`. |
| `data/agent-optimizer-dataset.jsonl` | Additional optimizer dataset stored with the project. |
| `agent-optimizer-baseline-results.jsonl` | Existing baseline evaluation output. |

## How it works

At startup, both scripts:

1. Load local settings from `.env`.
2. Authenticate to Microsoft Foundry with `DefaultAzureCredential`.
3. Load the agent instructions and structured data files.
4. Create a new version of the `risk-advisor-agent` prompt agent.
5. Attach Foundry file search and four local function tools:
   - claim history;
   - insurance program;
   - facility exposure; and
   - supply-chain exposure.
6. Configure client-side tracing when the Foundry project exposes an
   Application Insights connection string.

`agent.py` keeps one conversation open for an interactive chat session.
`agent-v2.py` creates a fresh conversation for every evaluation prompt so that
results are not affected by earlier examples.

## Prerequisites

- Python 3.9 or later
- Access to a Microsoft Foundry project
- A deployed model in that project
- A Foundry vector store containing the documents the agent should search
- An Azure identity that can access the project

`DefaultAzureCredential` supports several authentication methods. For local
development, signing in with the Azure CLI is typically the simplest:

```powershell
az login
```

## Local setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Update `.env` with values for your Foundry project:

| Variable | Required | Description |
| --- | --- | --- |
| `FOUNDRY_PROJECT_ENDPOINT` | Yes | Foundry project endpoint used by `AIProjectClient`. |
| `FOUNDRY_MODEL_DEPLOYMENT` | No | Model deployment name. The code has a default when this is omitted. |
| `VECTOR_STORE_ID` | Yes | Foundry vector store used by the hosted file-search tool. |
| `EVAL_MAX_RUNS` | No | Maximum evaluation rows processed by `agent-v2.py`; defaults to `40`. |
| `EVAL_DATASET_PATH` | No | Evaluation JSONL path; defaults to `data/agent-optimizer-eval.jsonl`. |
| `EVAL_RESULTS_PATH` | No | Results JSONL path; defaults to `agent-optimizer-baseline-results.jsonl`. |

The real `.env` file is intentionally ignored by Git. Keep project-specific
values and credentials there, and commit only `.env.example`.

## Run the interactive advisor

```powershell
python agent.py
```

Enter a question at the prompt. Use `exit` or `quit` to end the session. When
the model requests structured client information, the terminal shows the local
function tool calls before printing the final answer.

## Run the baseline evaluation

```powershell
python agent-v2.py
```

The runner reads JSONL rows containing at least `id` and `prompt`, executes each
prompt in a separate conversation, and writes results incrementally. Existing
content at the configured results path is replaced at the start of a run.

To run a smaller evaluation without editing source code, set `EVAL_MAX_RUNS`
in `.env`.

## Data model

The local JSON files are keyed by client name. A function tool returns the
matching object from its data source, or `null` when the requested client is
not present. Unstructured company information is retrieved separately by
Foundry's hosted file-search tool.

Use synthetic or approved data only. The sample data and evaluation results may
contain business-sensitive information and should be reviewed before sharing.
