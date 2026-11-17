# Finance Assistant

Python services for finance data, PDF-based knowledge retrieval, and MCP tool
access. The project uses a `src/` layout; run Python modules with `PYTHONPATH=src`
from the repository root.

## Architecture

```text
src/
├── agents/       # Six specialist definitions and workflow router
├── core/         # Settings, errors, and SQLite conversation/profile storage
├── data/         # Finance API, Qdrant adapters, and source PDFs
├── mcp_server/   # MCP transport and tool registration
├── api/          # FastAPI endpoints for the conversational and dashboard UI
├── rag/          # PDF parsing and dense/sparse embedding models
├── utils/        # Shared logging configuration
├── web_app/      # Streamlit frontend that calls the FastAPI service
└── workflow/     # Ingestion and LangGraph assistant workflow
tests/            # Unit tests
config.yaml       # Non-secret application defaults
requirements.txt  # Python dependencies
```

The MCP server exposes market data, portfolio analysis, and hybrid Qdrant
knowledge search. The LangGraph workflow routes each question to one of six
specialists: Finance Q&A, Portfolio Analysis, Market Analysis, Goal Planning,
News Synthesizer, or Tax Education. Each turn performs a routed RAG retrieval
before the specialist responds. The router and specialists use LangChain; the
specialists receive only their assigned MCP tools. Knowledge retrieval searches
the `text-dense` and
`text-sparse` vectors and fuses results with Qdrant Reciprocal Rank Fusion
(RRF). Search results are cached in memory for the configured TTL.

## High-Level Design (HLD)

### Purpose and scope

The application answers finance questions by routing each request to a
specialist, retrieving relevant PDF knowledge, and optionally calling external
finance tools through MCP. It provides a command-line assistant and a Streamlit
web UI. The UI calls the FastAPI service in `src/api/` for chat, portfolio
analysis, and market data.

## Run the API and Streamlit UI

From the repository root, create the virtual environment and install the
project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The Streamlit app is the browser UI and communicates with the backend through
FastAPI; it does not access the workflow or finance services directly. Start
MCP, FastAPI, and Streamlit in separate terminals:

```bash
# Terminal 1: MCP tools
PYTHONPATH=src .venv/bin/python -m mcp_server.server

# Terminal 2: FastAPI backend
PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8080

# Terminal 3: Streamlit frontend
PYTHONPATH=src .venv/bin/streamlit run src/web_app/app.py
```

Open `http://localhost:8501` for the UI. Its sidebar uses
`http://127.0.0.1:8080` as the FastAPI base URL by default; change it there if
the API is running elsewhere and use **Check API connection** to verify it.
The UI includes:

- **Chat**: sends questions to `POST /chat`, loads persisted conversation turns,
  and can clear the selected conversation.
- **Portfolio**: submits ticker, share quantity, and optional average cost to
  `POST /portfolio/analysis`; displays position values and allocation charts.
- **Market**: loads quote, company overview, and chart-ready daily price data
  from the `/market` endpoints.

Choose a distinct conversation ID to start a separate chat thread. User and
conversation IDs only scope stored data; they are not authentication. The API
health check is `http://127.0.0.1:8080/health`; interactive API documentation
is at `http://127.0.0.1:8080/docs`.

Set credentials in `.env` for the services you use. Chat requires
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and working Qdrant and MCP services.
Market-data routes require `ALPHA_VANTAGE_API_KEY`.

MCP uses port `8000`; the API uses `8080` to avoid conflict. Postman-friendly
OpenAPI docs are at `http://127.0.0.1:8080/openapi.json`.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the API is running. |
| `POST` | `/chat` | Send a message through the assistant workflow. |
| `GET` | `/chat/history` | Read saved turns for a user and conversation. |
| `DELETE` | `/chat/history` | Delete saved turns for a user and conversation. |
| `POST` | `/portfolio/analysis` | Analyze positions using current market quotes. |
| `GET` | `/market/quote/{symbol}` | Get the latest available quote. |
| `GET` | `/market/overview/{symbol}` | Get company fundamentals. |
| `GET` | `/market/history/{symbol}` | Get daily OHLCV history for charting. |

Example chat request:

```bash
curl --request POST 'http://127.0.0.1:8080/chat' \
  --header 'Content-Type: application/json' \
  --data '{
    "message": "Explain portfolio diversification.",
    "user_id": "local-user",
    "thread_id": "default"
  }'
```

Chat `user_id` and `thread_id` default to `local-user` and `default`.
Conversation history is persisted locally and recent turns are included as
context in later requests in that thread. Use a different `thread_id` for a new
conversation. These IDs are for local data scoping, not authentication.

History endpoints accept `user_id` and `thread_id` query parameters, plus an
optional `limit` from 1 to 500 on `GET /chat/history`:

```bash
curl 'http://127.0.0.1:8080/chat/history?user_id=local-user&thread_id=default&limit=20'
curl --request DELETE 'http://127.0.0.1:8080/chat/history?user_id=local-user&thread_id=default'
```

Portfolio analysis accepts 1–50 holdings. Each holding needs a bare ticker
symbol and a positive share quantity; `average_cost` is optional:

```bash
curl --request POST 'http://127.0.0.1:8080/portfolio/analysis' \
  --header 'Content-Type: application/json' \
  --data '{
    "holdings": [
      {"symbol": "AAPL", "quantity": 10, "average_cost": 150},
      {"symbol": "MSFT", "quantity": 5}
    ]
  }'
```

The response includes each position's current price, market value, and
allocation percentage, plus portfolio totals and any available gain/loss data.
It is an informational summary, not investment advice.

Market-data examples:

```bash
curl 'http://127.0.0.1:8080/market/quote/AAPL'
curl 'http://127.0.0.1:8080/market/overview/AAPL'
curl 'http://127.0.0.1:8080/market/history/AAPL?outputsize=compact'
```

The history `outputsize` can be `compact` or `full`. The API returns normalized
daily price points (`date`, `open`, `high`, `low`, `close`, and `volume`) ordered
oldest to newest for charting.

For debugging, set `API_DEBUG=true` in `.env` (or `api.debug: true` in
`config.yaml`) and restart the API. This enables FastAPI debug tracebacks and
request method, path, status, and duration logging. In VS Code, choose **Python:
Debug API** to debug API calls, or **Python: Debug API (with MCP)** to start both
services. Both launch profiles enable API debug mode; API listens on `8080`,
MCP on `8000`. Keep debug mode disabled in production because tracebacks can
disclose internal details.

Browser origins are restricted to common local development ports by default.
Set `FINANCE_ASSISTANT_CORS_ORIGINS` to a comma-separated allowlist for a
deployed UI. Do not expose the API publicly without authentication or a trusted
gateway.

### System context

```mermaid
flowchart LR
    User[User] --> Streamlit[Streamlit web UI]
    Streamlit -->|HTTP API calls| API[FastAPI service]
    API --> Workflow[LangGraph assistant workflow]
    API --> ConversationStore[SQLite conversation and profile store]
    API --> FinanceAPI[Market and portfolio endpoints]
    User --> CLI[Interactive CLI]
    CLI --> Workflow
    Workflow -->|user_id, thread_id| ConversationStore[SQLite conversation and profile store]
    ConversationStore -->|recent turns| Workflow
    ConversationStore -->|explicit learning profile| Workflow
    Workflow --> Router[OpenRouter JEV intent model]
    Router --> Specialist[Selected finance specialist]
    Workflow --> MCP[MCP Streamable HTTP server]
    Specialist --> MCP
    MCP --> Knowledge[Hybrid Qdrant knowledge service]
    Knowledge --> Qdrant[(Qdrant collection)]
    Knowledge --> Embedding[FastEmbed dense and sparse models]
    MCP --> AlphaVantage[Alpha Vantage API]
    FinanceAPI --> AlphaVantage
    Workflow --> LLM[OpenAI chat model]
    Specialist --> LLM
    Ingestion[PDF ingestion workflow] --> PDFs[Finance Q&A, news, tax PDFs]
    Ingestion --> Embedding
    Ingestion --> Qdrant
    Workflow -. "optional run traces" .-> LangSmith[LangSmith tracing]
    EvalSet[Curated evaluation dataset] -. "A/B experiments and evaluators" .-> LangSmith
```

### Low-Level Design: Settings loading

`src/core/settings.py` loads application configuration and exposes it as a
cached, immutable `Settings` object. Non-secret defaults come from
`config.yaml`; supported environment variables take precedence. The project
`.env` is loaded at module import and supplies environment values when they
aren't already set by the process.

```mermaid
flowchart TD
    A[Import core.settings] --> B[Resolve project root and config.yaml path]
    B --> C[Load project .env into process environment]
    C --> D[Application calls get_settings]
    D --> E{Cached Settings available?}
    E -- Yes --> F[Return cached Settings]
    E -- No --> G[Read and parse config.yaml]
    G --> H{Root is a YAML mapping?}
    H -- No --> X[Raise configuration or YAML error]
    H -- Yes --> I[Read named sections]
    I --> J[For each setting, prefer environment value over YAML default]
    J --> K[Parse integer and boolean values]
    K --> L{Values valid?}
    L -- No --> Y[Raise ValueError]
    L -- Yes --> M[Resolve relative data and database paths from project root]
    M --> N[Create frozen Settings dataclass]
    N --> O[Cache and return Settings]
```

Implementation details:

1. `PROJECT_ROOT` is derived from the location of `settings.py`; `CONFIG_FILE`
   points to `config.yaml` at the project root.
2. `_read_config()` reads YAML and rejects unreadable files, invalid YAML, or a
   root value that isn't a mapping. `_section()` similarly validates each
   configuration section.
3. `get_settings()` uses helper functions to read strings, integers, and
   booleans. Invalid integer/boolean environment values raise errors instead of
   silently falling back to defaults.
4. Relative `DATA_DIRECTORY` and `CONVERSATION_DB_PATH` values are resolved
   against `PROJECT_ROOT`. Secret values, such as API keys, are read from the
   environment or `.env`, not from `config.yaml`.
5. The returned `Settings` dataclass is frozen, and `@lru_cache(maxsize=1)`
   keeps one instance for the process. Tests call `get_settings.cache_clear()`
   when verifying different configurations.

### Main runtime flow

1. The CLI identifies the request with a `user_id` and `thread_id`. These IDs
   select a profile and a conversation; they are labels, not authentication.
2. Before invoking the graph, the workflow loads the user's explicit profile
   and the configured number of latest turns for that thread from SQLite.
   The full transcript remains stored.
3. The dedicated Jev intent model receives the current question and recent
   conversation turns through OpenRouter's Decisions API. The workflow selects
   the highest-scoring specialist. The user's profile is supplied to the
   selected specialist as context for adapting explanations and goals-based
   guidance.
4. The workflow retrieves relevant Qdrant chunks using the current question
   and recent conversation context, filtered by the selected specialist's
   document type. Each excerpt includes a source ID and available document
   title, explicit PDF filename, page range, and section metadata. If an older
   Qdrant point lacks `source_file`, the workflow uses its `DocType` to identify
   the configured PDF used for ingestion. Retrieval prompts use the current
   question alone unless it is a clear follow-up, when only the latest prior
   question is added; retrieved context is limited to four excerpts of at most
   1,800 characters each.
5. The specialist combines the question, retrieved knowledge, recent turns,
   and profile context, citing knowledge-base claims with the provided source
   IDs. The response's Sources list includes only retrieved PDF excerpts cited
   by the answer, and identifies any external API used by a successful
   finance-tool call. Retrieval results that the answer does not cite are not
   presented as answer sources.
   The prompt includes at most two recent conversation turns, with each
   message capped at 1,200 characters. Its LangChain agent can call only its
   assigned MCP tools; the MCP server executes those finance or knowledge
   operations.
6. After a successful response, the workflow stores the question and answer
   under that user and thread in SQLite. Failed requests are not added as
   completed turns.
7. The user can view conversation history and clear a conversation from either
   the API/Streamlit UI or CLI, and can manage their profile using CLI commands.
   Profile data is limited to learning level, explanation preferences, and
   financial goals; holdings and account details are not automatically retained.

The local interactive CLI resumes history when the same user and conversation
IDs are supplied after restarting. Choosing another conversation ID starts a
separate thread, while the user's profile remains shared across their threads.
The graph selects one specialist per question and does not fan out to multiple
specialists. The Streamlit UI is a separate frontend that communicates with the
assistant and finance endpoints through FastAPI.

### LangGraph workflow

`answer_question` loads the user's profile and recent conversation turns from
SQLite before invoking the compiled graph. The graph routes the question,
retrieves context for the selected specialist, and then runs only that
specialist. On success, the answer and question are saved to the same
conversation.

```mermaid
flowchart TD
    START([START]) --> ROUTER[route_query<br/>Intent router]
    ROUTER --> RETRIEVE[retrieve_context<br/>MCP search_knowledge]
    RETRIEVE --> ROUTE{Selected specialist}

    ROUTE -->|finance_qa| FINANCE[Finance Q&amp;A]
    ROUTE -->|portfolio_analysis| PORTFOLIO[Portfolio Analysis]
    ROUTE -->|market_analysis| MARKET[Market Analysis]
    ROUTE -->|goal_planning| GOALS[Goal Planning]
    ROUTE -->|news_synthesizer| NEWS[News Synthesizer]
    ROUTE -->|tax_education| TAX[Tax Education]

    FINANCE --> END([END])
    PORTFOLIO --> END
    MARKET --> END
    GOALS --> END
    NEWS --> END
    TAX --> END

    HISTORY[(SQLite conversation store)] -. load history and profile .-> ROUTER
    FINANCE -. generated answer .-> HISTORY
    PORTFOLIO -. generated answer .-> HISTORY
    MARKET -. generated answer .-> HISTORY
    GOALS -. generated answer .-> HISTORY
    NEWS -. generated answer .-> HISTORY
    TAX -. generated answer .-> HISTORY
    RETRIEVE -. search_knowledge .-> MCP[MCP server<br/>knowledge and finance tools]
    MCP -. filtered hybrid search .-> QDRANT[(Qdrant knowledge base)]

    PORTFOLIO -. optional assigned tools .-> MCP
    MARKET -. optional assigned tools .-> MCP
    GOALS -. optional assigned tools .-> MCP
    NEWS -. optional assigned tools .-> MCP
```

The dashed edges show supporting operations rather than extra LangGraph nodes:
conversation/profile reads happen before graph invocation, the final answer is
persisted after successful completion, retrieval calls Qdrant through MCP, and
only specialists with assigned finance tools can call those tools.

Conversation history and profile data are stored locally in the configured
SQLite database, independently from LangSmith traces. When tracing is enabled,
the model inputs can include recent turns and profile context; operators
should account for this data flow in their privacy and retention policies.

### Evaluation and observability

The LangGraph workflow supports optional LangSmith tracing through LangChain's
environment-based tracing integration. Set `LANGSMITH_TRACING=true` and provide
`LANGSMITH_API_KEY` to send runs to LangSmith. The root workflow run is named
`finance-assistant` and tagged `finance-assistant`; child graph nodes, model
calls, and tool calls are included in the trace. `LANGSMITH_PROJECT` selects
the destination project; the supplied `.env.example` sets it to
`finance-assistant`.

Offline evaluation should run separately from the interactive request path
against a curated, versioned dataset of finance questions and expected
behavior. Evaluators should measure:

- **Routing accuracy:** whether the router selects the intended specialist.
- **Retrieval quality:** whether retrieved chunks are relevant to the question
  and selected document type.
- **Answer quality:** relevance, groundedness in retrieved context, and
  appropriate handling of missing evidence.
- **Tool behavior:** whether the selected specialist invokes appropriate MCP
  tools with valid arguments when required.

Evaluation results can be compared across prompt, model, retrieval, and
workflow changes and used as a release-quality check. They should not block or
add latency to normal user requests. Traces can contain the user's question,
retrieved content, model output, and tool inputs/outputs. Do not send
personally identifiable information or financial account data to LangSmith
unless the project's privacy, consent, and retention policies permit it.

The evaluation harness creates or reuses the versioned
`finance-assistant-routing-v1` LangSmith dataset from
`src/evaluation/datasets/finance_router_v1.json`. It has twelve synthetic
queries covering all six specialists. Four evaluators record exact route
accuracy, non-empty answers, source-attribution completeness (including both
PDF and external API sources when present), and rubric-based answer quality
from the configured OpenAI model. Experiment A uses the Jev Decisions API
router; experiment B uses an OpenAI structured-output router. Both use the same
specialist answer model, retrieval, MCP tools, dataset, and evaluators.

The initial 12-example run on 2026-10-29 completed both experiments. Both
variants achieved 100% route accuracy and 100% non-empty answers. The normalized
LLM-judge answer-quality scores were 0.850 for Jev and 0.883 for OpenAI. This
small synthetic dataset is a smoke-test baseline, not sufficient evidence that
one router is generally better; rerun after changes and expand the dataset for
release decisions.

- [Experiment A: Jev Decisions API](https://smith.langchain.com/o/3e206f05-13ed-4e56-bfc2-974149837580/projects/p/8eb1c334-c3b9-4bfd-bf23-af0943d8b982)
- [Experiment B: OpenAI structured router](https://smith.langchain.com/o/3e206f05-13ed-4e56-bfc2-974149837580/projects/p/38f47de2-d292-49f2-abbc-e9099f018f61)
- [Compare the two experiments](https://smith.langchain.com/o/3e206f05-13ed-4e56-bfc2-974149837580/datasets/6d6145db-993c-4304-b89a-1699a0d78e9c/compare?selectedSessions=8eb1c334-c3b9-4bfd-bf23-af0943d8b982%2C38f47de2-d292-49f2-abbc-e9099f018f61)

Run both experiments from the project root after starting the MCP server and
setting `LANGSMITH_API_KEY`, `OPENAI_API_KEY`, and `OPENROUTER_API_KEY` in the
local `.env`:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.langsmith_experiments
```

To create or reuse the LangSmith dataset without running model calls:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.langsmith_experiments --dataset-only
```

The script prints experiment URLs and a comparison URL. It runs one example at a
time to limit concurrent calls to external model and MCP services. Use only
synthetic evaluation queries; LangSmith traces can retain prompts, retrieved
content, tool inputs/outputs, and model answers. This baseline comparison
measures routing impact while keeping answer generation and supporting services
constant; market/news examples can still depend on external data availability.
Each run creates new experiment results; existing LangSmith experiments are not
re-evaluated when evaluator code changes. Run the command again to see the
`source_attribution` metric on new experiment rows.

### Deployment boundaries

- **Assistant workflow:** runs separately from the MCP server and connects to
  the configured Streamable HTTP endpoint (`MCP_SERVER_URL`).
- **MCP server:** exposes market-data, portfolio, and knowledge-search tools at
  `http://127.0.0.1:8000/mcp` by default.
- **Qdrant:** stores PDF chunks, payload metadata, and named dense/sparse
  vectors.
- **External services:** Alpha Vantage supplies market data; OpenAI supplies
  router and specialist model inference. LangSmith is an optional tracing
  service enabled with environment settings.
- **Ingestion:** a separately invoked batch workflow parses local PDFs,
  computes embeddings, and upserts points into Qdrant.

## Low-Level Design (LLD)

### Component responsibilities

| Component | Responsibility |
|---|---|
| `core/settings.py` | Loads `config.yaml`, `.env`, and environment overrides into cached typed settings. |
| `core/conversation_store.py` | Persists conversation turns and explicitly maintained learning profiles in SQLite. |
| `core/errors.py` | Defines the shared application exception base. |
| `api/main.py` | Exposes health, chat/history, portfolio analysis, and market-data FastAPI endpoints; lazily builds and shares the assistant graph. |
| `web_app/app.py` | Provides the Streamlit chat, portfolio, and market UI by calling the FastAPI endpoints. |
| `agents/router.py` | Defines the structured route decision and routing instructions. |
| `agents/specialists.py` | Defines the six specialist identities, prompts, knowledge filters, and allowed MCP tool names. |
| `workflow/assistant.py` | Loads recent history and user profile, routes intent with the configured OpenRouter model, runs retrieval and specialist execution with the answer model, persists successful turns, and exposes the interactive CLI. |
| `mcp_server/server.py` | Registers MCP tools and serves them using Streamable HTTP. |
| `data/finance_api.py` | Validates finance API inputs, calls Alpha Vantage with retries, caching, and TLS verification; implements portfolio summary. |
| `data/knowledge_service.py` | Embeds query text, performs dense and sparse Qdrant prefetch, fuses ranks with RRF, and caches search results. |
| `rag/embedding.py` | Lazily initializes FastEmbed dense and SPLADE sparse models. |
| `rag/pdf_ingestion.py` | Parses and chunks finance, news, and tax PDFs with section metadata. |
| `workflow/ingestion.py` | Embeds PDF chunks, builds Qdrant points, creates the collection when missing, and uploads in batches. |
| `data/qdrant_cleanup.py` | Counts collection points by default and deletes only when explicitly confirmed. |

### LangGraph state and routing

The graph state is `AssistantState`:

| Field | Type | Meaning |
|---|---|---|
| `query` | `str` | User question for this invocation. |
| `route` | `str` | Selected `AgentType` value. |
| `retrieved_context` | `str` | Serialized result from the `search_knowledge` MCP tool. |
| `answer` | `str` | Specialist's final response. |
| `conversation_history` | `list[dict[str, str]]` | Recent user/assistant turns for the current user and conversation. |
| `user_profile` | `dict[str, Any]` | Explicit learning level, explanation preferences, and financial goals. |

Graph topology:

```text
START → router → retrieve_context → selected specialist → END
```

The router returns a `RouteDecision` with an agent enum value and rationale.
The retrieval node maps that agent to one or more `DocType` filters. Tool-enabled
specialists use LangChain `create_agent`; Finance Q&A and Tax Education call
the chat model directly after retrieval because they have no additional
specialist-specific MCP tools. All specialists still use the shared retrieval
tool before answering.

### Conversation history and user profiles

`core/conversation_store.py` stores successful user/assistant turns and
user-maintained profiles in `data/finance_assistant.sqlite3` by default.
`storage.sqlite_path` and `storage.history_turns` in `config.yaml` configure
the database location and how many recent turns are sent to the model. The
complete conversation remains stored; only the configured recent turns are
loaded into model context. Each turn is isolated by both `user_id` and
`thread_id`; profiles are isolated by `user_id`.

The profile contains only the fields the user explicitly enters: learning
level, explanation preferences, and financial goals. Holdings and account
details are not collected into the profile. In the CLI, `/profile` displays
the current profile, `/profile set` edits it, and `/profile delete` removes it.
`/history` displays the current conversation; `/clear-history` removes only
that conversation's turns. Reuse the same IDs to resume history; choose a new
conversation ID to begin a separate thread.

The CLI's IDs are organizational labels, not authentication. This local SQLite
implementation is intended for a single-user machine and does not provide
multi-user access control or database encryption. The database file is created
with owner-only permissions on supported systems and is ignored by Git. When
LangSmith tracing is enabled, model context can include prior turns and profile
fields, so consider disabling tracing for sensitive conversations.

### Specialist knowledge and tool mapping

| Specialist | Qdrant `DocType` filter | Additional MCP tools |
|---|---|---|
| Finance Q&A | `QAndA` | None |
| Portfolio Analysis | `QAndA` | `analyze_portfolio` |
| Market Analysis | `NewsSynthesiser`, `QAndA` | `get_stock_quote`, `get_company_overview`, `get_daily_stock_prices` |
| Goal Planning | `QAndA`, `TaxEducation` | `analyze_portfolio`, `get_stock_quote`, `get_company_overview` |
| News Synthesizer | `NewsSynthesiser` | `get_financial_news` |
| Tax Education | `TaxEducation` | None |

The tool list is loaded dynamically from MCP when the graph is built. Startup
fails with an explicit error if the required retrieval tool or a specialist's
declared tool is unavailable. The MCP server registers functions decorated with
`@mcp.tool()` in `src/mcp_server/server.py`. The workflow discovers those tools
from the configured MCP server and maps them by name; each specialist's
`mcp_tools` frozenset in `src/agents/specialists.py` declares the additional
tools that specialist is permitted to use. The retrieval tool
`search_knowledge` is shared by the workflow and is required when the graph is
built.

### Hybrid retrieval and RRF

`data/knowledge_service.py` embeds the normalized query with the same dense and
sparse embedding helpers used during document ingestion. It issues two
prefetches against Qdrant's named vectors:

- `text-dense`: dense vector similarity.
- `text-sparse`: SPLADE sparse vector similarity.

The candidates are fused with Qdrant's RRF query and then limited to the
requested result count. Optional `doc_types` are applied as a Qdrant `DocType`
payload filter. Returned results contain the chunk text, fused score, and
metadata; internal embedding-input text is excluded from returned metadata.

Searches use a process-local LRU-style cache with a configurable TTL and
maximum number of entries. Cache keys include collection, query, document-type
filter, and result limit. Cached values are deep-copied to prevent callers from
mutating shared cache entries. The cache is not shared across server processes.

### Qdrant document schema

Each ingested point has a deterministic UUID, two named vectors, and a payload:

```text
Point
├── id: UUIDv5(document identity)
├── vector
│   ├── text-dense: 1024-dimensional vector
│   └── text-sparse: {indices, values}
└── payload
    ├── DocType: QAndA | NewsSynthesiser | TaxEducation
    ├── combined_text: retrieved chunk
    └── document, section, keyword, and page metadata
```

The ingestion workflow creates the `FinAi` collection if absent, configures
cosine distance for the 1024-dimensional dense vector, configures a sparse
vector index, creates a keyword index for `DocType`, and upserts chunks in
batches. Existing collections are not recreated or automatically migrated.

### Configuration and secrets

`core/settings.py` loads non-secret defaults from `config.yaml` and lets
environment variables override supported values. Secrets are read from
environment variables or the local, ignored `.env` file; they must not be
stored in `config.yaml` or committed. Important settings include:

- Model and MCP endpoint: `OPENAI_MODEL`, `MCP_SERVER_URL`.
- Specialist response model: `OPENAI_API_KEY`, `OPENAI_MODEL`.
- Specialist intent router: `OPENROUTER_API_KEY`; the model and API base URL
  are set under `router` in `config.yaml` and can be overridden with
  `OPENROUTER_MODEL` and `OPENROUTER_BASE_URL`.
- Credentials: `OPENAI_API_KEY`, `ALPHA_VANTAGE_API_KEY`.
- Optional tracing: enable `langsmith.tracing` in `config.yaml`, or override
  it with `LANGSMITH_TRACING`; provide `LANGSMITH_API_KEY` in `.env`.
  Project and endpoint defaults are configured under `langsmith` in
  `config.yaml` and can be overridden with `LANGSMITH_PROJECT` and
  `LANGSMITH_ENDPOINT`.
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
- Embeddings: `DENSE_EMBEDDING_MODEL`, `SPARSE_EMBEDDING_MODEL`.
- Retrieval cache and limits: `SEARCH_CACHE_TTL_SECONDS`,
  `SEARCH_CACHE_MAX_ENTRIES`, `SEARCH_MAX_RESULTS`.
- Conversation persistence: `CONVERSATION_DB_PATH`,
  `CONVERSATION_HISTORY_TURNS`.
- API debugging: `API_DEBUG` or `api.debug` in `config.yaml`; disabled by
  default. Debug tracebacks can disclose internal details, so leave it disabled
  in production.

### Reliability, security, and limitations

- Alpha Vantage requests use verified TLS, bounded retries, timeouts, input
  validation, and response caching.
- Qdrant and API errors are raised with operation-specific context rather than
  silently returning empty success-shaped results.
- Logging is configured to write to the console; the application does not
  currently configure persistent log files.
- Finance responses are informational, not investment, legal, or tax advice.
- MCP transport defaults to local Streamable HTTP. Production deployments
  should configure authentication, network controls, and TLS termination.
- Current unit tests cover workflow routing/retrieval, tool assignment,
  hybrid RRF query construction/cache behavior, settings overrides, and
  ingestion point conversion. Live OpenAI, Alpha Vantage, and Qdrant integration
  tests require explicit service credentials and are not run by the unit suite.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Add credentials and service URLs to `.env`:

- `ALPHA_VANTAGE_API_KEY` for market-data tools.
- `QDRANT_URL` and `QDRANT_API_KEY` for knowledge search and ingestion.
- `OPENAI_API_KEY` for specialist responses. `OPENAI_MODEL`
  selects the specialist response model.
- `OPENROUTER_API_KEY` for the dedicated intent router. The default router
  model is `~typesafe/jev-latest`; configure `router.model` in `config.yaml` or
  `OPENROUTER_MODEL` if your OpenRouter account uses a different model ID.
- Optional LangSmith tracing: set `langsmith.tracing: true` in `config.yaml`
  and provide your `LANGSMITH_API_KEY` in `.env`. The config sets project and
  endpoint defaults; environment variables can override them. Tracing is
  disabled by setting `langsmith.tracing: false` or `LANGSMITH_TRACING=false`.

Never commit `.env` or real credentials. Non-secret defaults, such as model
names, collection name, retrieval limits, and logging level, live in
`config.yaml`. Environment variables override supported settings.

## Run the MCP server

```sh
PYTHONPATH=src .venv/bin/python -m mcp_server.server
```

The Streamable HTTP endpoint is `http://127.0.0.1:8000/mcp`. The server remains
running while clients connect. In VS Code, start the **Python: Debug MCP
server** launch configuration, then connect to the `finance-assistant` server
in `.vscode/mcp.json`.

### Run a local concurrent load test

Install the development dependency once:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
```

Start the MCP server in one terminal. In another, run Locust against the actual
assistant workflow:

```sh
PYTHONPATH=src LANGSMITH_TRACING=false \
MCP_SERVER_URL=http://127.0.0.1:8000/mcp \
CONVERSATION_DB_PATH=data/locust_load_test.sqlite3 \
.venv/bin/locust -f load_test/locustfile.py --headless \
  --html load_test/report.html --csv load_test/results \
  --users 20 --spawn-rate 5 --run-time 1m
```

Each virtual user asks the workflow a question using a unique user ID and
conversation thread. Locust reports end-to-end `answer_question` latency and
failures, including routing, knowledge retrieval, and answer generation. The
test requires `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, a running MCP server, and
working Qdrant configuration. The default question is an educational prompt;
set `LOCUST_WORKFLOW_QUESTION` to test another workflow path. Finance-data
questions may also call Alpha Vantage.

The compiled graph is shared by concurrent requests on a dedicated asyncio
loop. Conversation history is stored in the separate local database configured
above. Tracing is disabled for this run to avoid sending test prompts to
LangSmith. Adjust `--users`, `--spawn-rate`, and `--run-time` for the desired
load.

The Locust scenario and its output files are kept together in `load_test/`.

## Run the LangGraph assistant

Start the MCP server first, then in another terminal run:

```sh
PYTHONPATH=src .venv/bin/python -m workflow.assistant
```

In VS Code, select **Python: Debug assistant** in the Run and Debug panel to
launch this same module with breakpoints. Start the MCP server separately
before debugging the assistant.

The CLI asks for a user ID and conversation ID; accept the defaults to resume
the default local conversation, or use distinct IDs for separate users and
threads. It then accepts questions until `/quit`. Use `/profile`, `/profile set`,
`/profile delete`, `/history`, and `/clear-history` to manage saved data.
The workflow connects to `MCP_SERVER_URL`, retrieves relevant knowledge, and
uses recent conversation history for intent routing and specialist context,
and uses the explicit profile for specialist personalization. Market,
portfolio, and news specialists may call only their assigned MCP tools. It
requires `OPENAI_API_KEY` and `OPENROUTER_API_KEY`; data-dependent MCP tools
also require their API/Qdrant configuration.

### Manually test persistent conversation history

1. Start the MCP server and assistant. When prompted, use a test user ID such
   as `history-test-user` and conversation ID `history-test-thread`.
2. Ask: `I am learning about retirement accounts. Explain a Roth IRA simply.`
   Wait for the assistant's answer.
3. Ask a follow-up that relies on the earlier turn:
   `Compare that with a traditional IRA, and remind me which account I asked about first.`
   The answer should use the current question and earlier conversation context.
4. Enter `/history`. Both completed question/answer pairs should be listed.
5. Enter `/quit`, restart the assistant, and enter the **same** user and
   conversation IDs. Enter `/history`; the turns should still be there. Ask
   another follow-up to check that the prior turns are passed back as context.
6. Quit and restart with the same user ID but a different conversation ID.
   `/history` should report no saved turns for that thread.

To check persistence and isolation without making model or MCP calls, run the
store unit tests:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_conversation_store -v
```

The test verifies that turns survive reopening the SQLite database, stay
scoped to the requested user and conversation IDs, and can be cleared per
conversation. The workflow tests also check that prior turns and explicit
profile fields are included in model context.

## Ingest the PDF knowledge base

The ingestion workflow reads `financeQnA.pdf`, `News_Systhesizer.pdf`, and
`tax_education.pdf` from `src/data/`, creates embeddings, and upserts chunks
into Qdrant. It can create the configured collection if it does not exist.

```sh
PYTHONPATH=src .venv/bin/python -m workflow.ingestion
```

To count points before deleting them, run:

```sh
PYTHONPATH=src .venv/bin/python -m data.qdrant_cleanup
```

Deletion requires the explicit `--confirm` option.

## Tests

Run the unit tests without external API requests or Qdrant credentials:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
