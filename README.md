# Multi-Agentic Voice AI — Nawaloka Hospital

> **Production-shaped voice assistant** — LiveKit + Deepgram STT + ElevenLabs TTS + Groq llama-3.3-70b sit on top of a LangGraph multi-agent system with a 4-tier memory, MCP-backed CRM, RAG knowledge base, real-time web search, Langfuse observability, and a reactive React UI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-teal.svg)](https://fastapi.tiangolo.com/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents%201.5-red.svg)](https://docs.livekit.io/agents/)
[![Deepgram](https://img.shields.io/badge/Deepgram-Nova--3-black.svg)](https://deepgram.com/)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Turbo%20v2.5-yellow.svg)](https://elevenlabs.io/)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-lightgrey.svg)](https://groq.com/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange.svg)](https://langfuse.com/)

---

## What's new in Week 15

This week the voice path moved from *"works in LiveKit's playground"* to *"first-class feature in the actual app"* — with a measured sub-2-second latency budget, integrity-preserving memory on barge-in, and full-stack observability.

| Feature | Impact |
|---|---|
| **Voice fast path** (`achat_stream_fast`) | Bypasses the multi-agent graph for voice — single streaming LLM call. Drops perceived latency from ~32 s → ~1.5 s. |
| **Real token streaming** (`streaming=True` on the fast LLM) | First-token latency 2247 ms → 379 ms. Real streaming, not client-side chunking. |
| **Memory + barge-in integrity** | If the user interrupts mid-sentence, the *partial* answer (what was actually spoken) is saved to memory tagged `[interrupted]`. Long-term distillation skips interrupted turns. The agent's recollection matches the user's experience. |
| **Reactive UI bubble** (`VoiceBubble.tsx` + `VoiceRoom.tsx`) | WebAudio-driven SVG that pulses with mic + agent audio. Five states: idle / listening / thinking / speaking / error. Latency HUD beneath. |
| **LiveKit token endpoint** (`/voice/token`) | Browser-callable JWT minter so the UI joins LiveKit rooms without exposing the API secret. |
| **Sidebar split — Voice / Chat** | The session list now has two halves: voice calls on top, text chats below. Client-side partition by `voice-` prefix on `session_id`. |
| **Auto-generated session titles** | After 4 turns, a fast Groq LLM summarises the conversation into a 3–6 word title. Replaces `"Conversation 2026-05-17 12:30"` with something like `"Cardiology Appointment Booking"`. Costs ~$0.00001 per session. |
| **Latency observability triangle** | Per-turn timings visible in **three places**: worker log, Langfuse `voice_pipeline` span with metadata, browser HUD via LiveKit data channel. |
| **Tuned VAD endpointing** | `silence_threshold_ms`: 500 → 300, `min_endpointing_delay`: 0.5 → 0.3. 400 ms saved per turn. Yaml + dataclass defaults now in sync. |
| **Worker pre-warming** (`warm_start`) | One tiny `llm_fast.ainvoke("hi")` at boot primes the HTTPS pool so call #1 doesn't pay TLS handshake cost. |
| **Langfuse v3+/v2 import fallback** | Tracing works whether your env has langfuse v2 or v4. |

---

## Architecture

Two entry surfaces, one orchestrator core.

```mermaid
flowchart TD
    subgraph TEXT["TEXT PATH (Week 13)"]
        B[Browser] --> FA["FastAPI /chat"]
        FA --> DG[decision_graph]
        DG --> GR["guardrail (Llama)"]
        GR --> CAG["CAG cache (Qdrant)"]
    end
    subgraph VOICE["VOICE PATH (Week 14 + 15)"]
        BR["Browser (React)"] --> TOK["POST /voice/token"] --> JWT[JWT]
        JWT --> LKC["livekit-client.connect()"]
        LKC --> LK[LiveKit Cloud]
        LK --> VW[Voice Worker]
        VW --> VAD[Silero VAD] --> STT[Deepgram STT] --> ADP[LangGraphLLMAdapter]
        ADP --> FAST["orchestrator.achat_stream_fast()  ← Week 15"]
        FAST --> GROQ["Groq llama-3.3-70b (streaming, real)"]
        GROQ --> TOK_OUT[tokens] --> TTS["ElevenLabs TTS (streaming)"] --> AUD[audio]
        GROQ --> BG[bg: _save_voice_turn_async]
        BG --> TS[touch_session<br/>sidebar row]
        BG --> AT[maybe_auto_title<br/>LLM]
        BG --> DIST["distill (if not interrupted)"]
        GROQ -.data channel.-> HUD[latency HUD]
        HUD --> BUB[Browser bubble]
    end
    CAG --> ACHAT["orchestrator.achat()"]
    ACHAT --> AO["AgentOrchestrator (LangGraph)"]
    AO --> RECALL[recall] --> SUP[supervisor] --> FAN[fan-out]
    FAN -.-> ADM[admin]
    FAN -.-> CLIN[clinical]
    FAN -.-> DIR[direct]
    FAN -.-> WEB[web]
    ADM --> MERGE[merge_responses]
    CLIN --> MERGE
    DIR --> MERGE
    WEB --> MERGE
    MERGE --> SM[save_memory]
    SM --> MEM["4-tier Memory<br/>(ST / LT / EP / Pr)<br/>Supabase + pgvector<br/>+ Qdrant (RAG, CAG)"]
    MEM --> LFUSE["Langfuse<br/>(per-turn trace + Sessions view)"]
    BG --> LFUSE
```

**Key boundary:** `src/voice/` and `ui/src/components/Voice*.tsx` are a self-contained vertical slice. The voice fast path is the only orchestrator addition (`achat_stream_fast`); the multi-agent text graph is untouched. Removing the voice layer leaves Week 13 fully functional.

### Voice pipeline internals (Week 15)

```mermaid
flowchart TD
    A["LiveKit Agent<br/>(livekit.agents.voice.Agent)"]
    A --> VAD["VAD: silero.VAD.load<br/>activation_threshold=0.5<br/>min_silence_duration=0.3 s"]
    A --> STT["STT: deepgram.STT<br/>(model=nova-3, language=en) — streaming"]
    A --> LLM["LLM: LangGraphLLMAdapter(orchestrator)"]
    LLM --> RUN["LangGraphLLMStream._run()"]
    RUN --> R1["open Langfuse 'voice_pipeline' span"]
    RUN --> R2["async for kind, payload in achat_stream_fast(...)"]
    R2 --> R2a["'token' → emit ChatChunk → TTS"]
    R2 --> R2b["'partial' → remember spoken (for barge-in)"]
    R2 --> R2c["'final' → AgentResponse with metadata"]
    RUN --> R3["on CancelledError (barge-in):"]
    R3 --> R3a["orchestrator saves [interrupted] partial"]
    R3 --> R3b["update Langfuse span (barge_in=true)"]
    R3 --> R3c["re-raise"]
    RUN --> R4["stash last_latency for data-channel publish"]
    A --> TTS["TTS: elevenlabs.TTS<br/>(model=eleven_turbo_v2_5,<br/>voice_id=l7kNoIfnJKPg7779LI2t) — streaming"]
    A --> OPTS["allow_interruptions=True<br/>min_endpointing_delay=0.3 s"]
```

**EOU policy** is three-layered:
1. VAD (`vad_threshold=0.5`) — *"is this speech right now?"*
2. Silence persistence (`silence_threshold_ms=300`)
3. Confirmation buffer (`min_endpointing_delay=0.3`)

Perceived endpoint = layer 2 + layer 3 = **600 ms**.

### Latency budget (measured)

```mermaid
flowchart TD
    U[User stops speaking] --> S1["VAD endpointing — 300 ms"]
    S1 --> S2["Deepgram STT (streaming) — 200 ms"]
    S2 --> S3["Orchestrator pre-LLM — &lt;150 ms<br/>(time-boxed memory fetch, off-thread)"]
    S3 --> S4["Groq llama-3.3-70b first token — 200–400 ms"]
    S4 --> S5["ElevenLabs TTS first byte — 200 ms"]
    S5 --> S6["Network (browser ↔ region) — 200 ms"]
    S6 --> TOTAL["~1.5 s perceived"]
```

The full Week 15 latency story includes a `streaming=False` bug fix that cost 2 seconds, and a sync Supabase fetch that cost another 700–1500 ms.

### MCP integration layer

```mermaid
flowchart TD
    O["orchestrator.py<br/>build_agent_mcp()"] --> MCL["MCP Client Layer<br/>(langchain-mcp-adapters)"]
    MCL --> S1[nawaloka-crm<br/>custom]
    MCL --> S2[nawaloka-memory<br/>custom]
    MCL --> S3[postgres MCP<br/>off-the-shelf]
    S1 --> T1[CRMTool]
    S2 --> T2[MemoryOps]
    S3 --> T3[Supabase raw SQL]
    T1 --> B1[Supabase]
    T2 --> B2[pgvector]
```

Three MCP servers, three origins, one agent. The text path uses MCP-backed tools; the voice fast path skips MCP for latency reasons and calls Groq directly.

---

## Project Structure

```
E2E Deployment/
│
├── src/
│   ├── voice/                                    # ← Week 14 voice side-car
│   │   ├── __init__.py
│   │   ├── config.py                             # ★ W15: VAD defaults 300/0.3
│   │   ├── stt.py                                # make_stt — Deepgram nova-3
│   │   ├── tts.py                                # make_tts — ElevenLabs / Deepgram
│   │   ├── adapter.py                            # ★ W15: LangGraphLLMStream._run rewritten
│   │   │                                         #         (token streaming + barge-in + Langfuse span)
│   │   ├── pipeline.py                           # VoiceSession + SessionManager + event helpers
│   │   ├── agent.py                              # ★ W15: warm_start wired, latency data-channel publish
│   │   └── run.py                                # ★ W15: initialize_process_timeout=60s
│   │
│   ├── agents/
│   │   ├── orchestrator.py                       # ★ W15: achat_stream_fast + _save_voice_turn_async
│   │   ├── decision_graph.py                     # Text-path guardrail + CAG short-circuit
│   │   ├── guardrail.py
│   │   ├── router.py
│   │   ├── state.py
│   │   ├── prompts/agent_prompts.py
│   │   └── tools/{crm_tool.py, rag_tool.py, web_search_tool.py}
│   │
│   ├── mcp_servers/                              # CRM, memory, RAG, web, CAG, crawler MCP servers
│   │
│   ├── api/
│   │   ├── main.py                               # ★ W15: voice_router registered
│   │   ├── schemas.py
│   │   └── routers/
│   │       ├── chat.py                           # ★ W15: maybe_auto_title_sync at 3 save sites
│   │       ├── chat_sessions.py                  # ★ W15: _is_default_title + maybe_auto_title_sync
│   │       ├── voice.py                          # ★ NEW W15: /voice/token JWT endpoint
│   │       ├── health.py, patients.py
│   │       └── tools/{cag,crawl,crm,memory,rag,web}.py
│   │
│   ├── memory/                                   # 4-tier memory (Week 13)
│   │   ├── st_store.py, lt_store.py
│   │   ├── episodic_store.py, procedural_store.py
│   │   ├── memory_ops.py                         # MemoryDistiller + MemoryRecaller
│   │   ├── schemas.py, prompts.py
│   │
│   ├── services/{chat_service, crm_service, ingest_service}/
│   │
│   └── infrastructure/
│       ├── config.py
│       ├── observability.py                      # ★ W15: v3+/v2 langfuse import fallback
│       ├── llm/
│       │   └── llm_provider.py                   # ★ W15: get_fast_chat_llm defaults streaming=True
│       ├── db/
│       └── log.py
│
├── ui/                                           # React + Vite + Tailwind + Framer Motion
│   └── src/
│       ├── App.tsx                               # ★ W15: Voice button + modal + sidebar-refresh hooks
│       ├── components/
│       │   ├── VoiceBubble.tsx                   # ★ NEW W15: reactive SVG blob (5 states)
│       │   ├── VoiceRoom.tsx                     # ★ NEW W15: LiveKit + WebAudio analysers
│       │   ├── Sidebar.tsx                       # ★ W15: split into Voice (top) + Chat (bottom)
│       │   ├── ChatWindow.tsx, InputBox.tsx, MessageBubble.tsx, …
│       │   └── …
│       └── hooks/{useChat, useChatStream, useSessions, useHealth, usePatient}.ts
│
├── notebooks/
│   ├── 01_routing_memory_and_tools.ipynb         # Week 13: 4-tier memory + routing
│   ├── 02_multi_agent_langgraph.ipynb            # Week 13: LangGraph multi-agent + MCP
│   ├── 03_voice_pipeline_fundamentals.ipynb      # Week 14: STT/TTS/VAD/EOU standalone
│   └── 04_voice_agent_livekit.ipynb              # Week 14: voice + LangGraph integration
│
├── docker/
│   ├── api/Dockerfile                            # FastAPI service
│   ├── web/Dockerfile                            # nginx + built React
│   └── voice/Dockerfile                          # LiveKit voice worker
│
├── scripts/
│   ├── seed_crm_unified.py, ingest_to_qdrant.py
│   ├── seed_procedures.py, rebuild_cag_cache.py
│   └── init_supabase.py
│
├── config/
│   └── param.yaml                                # ★ W15: VAD 300/0.3 defaults (yaml ↔ dataclass)
│
├── docs/
│   ├── week-15-16-deck.md                        # Source for the slide deck
│   └── _md_to_docx.py                            # Regenerates .docx from .md
│
├── Weeks_15–16_From_Local_Demo_to_Production_at_Scale-2.pdf   # ← the final slide deck
├── README.md                                     # ← this file
├── STUDENT_SETUP_GUIDE.md
├── Makefile                                      # demo / voice / voice-test / voice-logs / …
├── docker-compose.yml                            # api + web (default), voice (profile)
├── pyproject.toml                                # Source of truth for dependencies
├── requirements.txt                              # Lock-step with pyproject.toml
├── .env.example                                  # Template — includes voice section
└── uv.lock
```

**★ = files added or modified in Week 15.**

---

## Quick Start — Three Commands

```bash
# Terminal 1 — API + Web (Docker)
make demo

# Terminal 2 — Voice worker (foreground)
make voice

# Terminal 3 — UI (hot-reload dev server)
cd ui && npm install && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173` or `5174`) in **Chrome** (Safari has WebAudio quirks). Log in as a patient, click **Voice** in the top bar, click **Start Call**.

**Readiness signals:**
- API ready when log says `Application startup complete` (~60 s — lifespan startup is gated on Supabase + Qdrant + MCP + CAG warm-up).
- Voice worker ready when log says **`LLM connection warm — first call took XXX ms`** AND `registered worker`.
- UI ready when Vite prints the local URL.

If you hit `ModuleNotFoundError: No module named 'langchain_core'` running uvicorn natively, your shell's `python` resolves to homebrew's Python, not anaconda's. Use:
```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m uvicorn api.main:app --reload --port 8000
```

---

## Compose targets

| Command | Brings up | Containers |
|---|---|---|
| `make demo` | Default text stack | `api` + `web` |
| `make demo-voice` | Text stack + voice worker | `api` + `web` + `voice` |
| `make voice` | Voice worker only (native, foreground) | — |
| `make voice-test` | Validate voice config + env vars | — |
| `make demo-down` | Stop default stack | — |
| `make demo-voice-down` | Stop voice profile | — |
| `make demo-logs` / `make voice-logs` | Tail logs | — |

The voice worker has no exposed port — it dials outbound to `LIVEKIT_URL` and registers as a worker. Use `make voice-logs` to confirm registration.

**Try these to see each route light up:**

| Channel | Query | What it exercises | Latency |
|---|---|---|---|
| Text | `What are the opening hours?` | CAG cache → FAQ hit | ~290 ms |
| Text | `Do I have a booking next week?` | CRM → Supabase patient lookup | ~3–5 s |
| Voice | `Book me an appointment with a cardiologist` | Voice fast path | ~1.5 s |
| Voice | `Tell me about post-surgery care` *(interrupt mid-sentence)* | Barge-in + partial-answer memory | ~400 ms to silence |

---

## Voice configuration

`config/param.yaml`:

```yaml
voice:
  stt_provider: deepgram
  stt_model: nova-3
  stt_language: en

  tts_provider: elevenlabs            # or "deepgram"
  tts_model: eleven_turbo_v2_5
  tts_voice_id: l7kNoIfnJKPg7779LI2t  # ElevenLabs "Aria"

  # ── VAD + EOU policy ────────────────────────────────────────
  # Combined endpoint = silence_threshold_ms + min_endpointing_delay
  # 300 ms + 300 ms feels responsive without false interruptions.
  vad_threshold: 0.5
  silence_threshold_ms: 300           # was 500 — saved 200 ms per turn
  min_endpointing_delay: 0.3          # was 0.5 — saved 200 ms per turn

  interruption_enabled: true
  sample_rate: 16000
```

**Tuning knobs and their feel:**

| Knob | Lower | Higher |
|---|---|---|
| `vad_threshold` | More false positives (background noise becomes "speech") | Soft speakers get cut off |
| `silence_threshold_ms` | Snappy but interrupts people who pause | Polite but feels laggy |
| `min_endpointing_delay` | Fast turn-around | Better tolerates last-syllable trail-off |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send message, get reply (decision_graph → orchestrator). Background-schedules `touch_session` + `maybe_auto_title`. |
| `POST` | `/chat/stream` | SSE — node-by-node state updates. |
| `POST` | `/voice/token` | Mint a short-lived LiveKit JWT (10-min TTL) for the browser. **New in Week 15.** |
| `GET` | `/chat_sessions?user_id=…` | List a patient's sessions for the sidebar. Voice + chat sessions in one list, partitioned client-side by `voice-` prefix. |
| `POST` | `/chat_sessions` | Create a new chat session. |
| `PATCH` | `/chat_sessions/{id}` | Rename or archive. |
| `DELETE` | `/chat_sessions/{id}` | Hard-delete (cascades ST turns). |
| `GET` | `/sessions/{id}/turns` | Fetch ST turn history for a session. |
| `GET` | `/health` | Liveness check + tool availability. |
| `GET` | `/graph` | LangGraph topology (Mermaid + structured). |
| `GET` | `/memory/{user_id}` | Inspect long-term semantic facts. |

### Examples

```bash
# Text chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Who are the cardiologists?", "user_id": "94781030736", "session_id": "demo"}'

# Mint a voice token (the browser calls this)
curl -X POST http://localhost:8000/voice/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "94781030736"}'
# → {"url":"wss://...", "token":"eyJ...", "room":"voice-xxx", "identity":"94781030736"}

# List sessions for the sidebar
curl "http://localhost:8000/chat_sessions?user_id=94781030736"
```

The voice worker itself has no HTTP surface — it talks LiveKit's room protocol over WebRTC.

---

## Observability

Every `.chat()`, `.achat()`, and voice turn produces a Langfuse trace.

**Text path:**

```mermaid
flowchart TD
    T[trace: agent_chat] --> N1[node_recall<br/>ST + LT retrieval]
    T --> N2[node_supervisor<br/>router LLM generation]
    T --> N3["node_[agent]<br/>tool call + synthesis"]
    T --> N4[node_save_memory<br/>ST store + LT distillation]
```

**Voice path (Week 15):**

```mermaid
flowchart TD
    T[trace: voice_turn] --> SP["span: voice_pipeline<br/>user_id, session_id, tags=[voice, fast_path]"]
    SP --> M1[metadata.first_token_ms]
    SP --> M2[metadata.llm_total_ms]
    SP --> M3[metadata.agent_total_ms]
    SP --> M4[metadata.chunks]
    SP --> M5["metadata.barge_in (true if interrupted)"]
    SP --> GEN["generation: ChatGroq<br/>llama-3.3-70b-versatile"]
    GEN --> G1[input messages]
    GEN --> G2[output completion]
    GEN --> G3[token usage]
    GEN --> G4[cost]
```

All turns of one call share `session_id="voice-<room>"` → grouped in the Langfuse **Sessions** view. Filter by `tags:["voice","fast_path"]` for the voice subset.

Worker log shows per-turn timings in real time:
```
⏱  achat_stream_fast: mem=87ms, prompt=1ms, pre_llm_total=92ms, first_llm_chunk=287ms
📊 Voice turn timings: first_token=326ms, llm_total=510ms, agent_total=623ms, chunks=8, route=voice_fast
```

The same numbers flow to the browser HUD via the LiveKit data channel.

---

## MCP integration details

```python
# orchestrator.py — build_agent_mcp()

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_servers.mcp_config import build_mcp_server_config

mcp_client = MultiServerMCPClient(build_mcp_server_config())
tools = await mcp_client.get_tools()    # discovers tools from 3 servers
crm_tool = _MCPCRMToolAdapter(tools)    # same dispatch() interface
```

### MCP server architecture (stdio transport)

```mermaid
sequenceDiagram
    participant H as Host Process<br/>(LangGraph agent)
    participant S as CRM MCP Server<br/>(crm_server.py)
    H->>S: spawn subprocess
    H->>S: stdin (JSON-RPC request)
    S-->>H: stdout (JSON-RPC response)
    S-->>H: stderr (logs only)
```

> **Important:** Never `print()` inside an MCP server. stdout is reserved for the JSON-RPC protocol. Use `loguru` (defaults to stderr).

---

## External services

| Service | Purpose | Free tier |
|---|---|---|
| [Supabase](https://supabase.com) | PostgreSQL + pgvector (CRM + memory + chat_sessions) | Yes |
| [Qdrant Cloud](https://qdrant.tech) | Vector DB (RAG KB + CAG cache) | Yes (1 GB) |
| [Groq](https://groq.com) | llama-3.3-70b-versatile (voice fast path, router) | Pay-per-use |
| [OpenRouter](https://openrouter.ai) | Gemini 2.5 Flash (text synthesiser) | Pay-per-use |
| [Tavily](https://tavily.com) | Real-time web search | 1 000 free searches/mo |
| [Langfuse](https://langfuse.com) | Tracing, cost tracking, prompt versioning | Free (hobby) |
| [LiveKit Cloud](https://cloud.livekit.io) | WebRTC infra for voice rooms | Yes (dev tier) |
| [Deepgram](https://deepgram.com) | Streaming STT (Nova-3) | $200 credit |
| [ElevenLabs](https://elevenlabs.io) | Streaming TTS (Turbo v2.5) | 10 k chars/mo free |

---

## Course context

This codebase is the **Week 15** material for the AI Engineer Essentials bootcamp.

| Week | Topic | What it added |
|---|---|---|
| 6 | Agentic design patterns (from scratch) | The vocabulary |
| 7 | Memory + routing + multi-agent (from scratch) | Memory, classifier router |
| 9 | LangGraph fundamentals | StateGraph mental model |
| 10 | Multi-agent system (rebuilt on LangGraph) | Fan-out/fan-in topology |
| 12 | MCP integration (portable tools) | Tool boundary moves to MCP |
| 13 | Containerised + decision graph | Docker, FastAPI, React UI, guardrail, CAG |
| 14 | Voice interface | LiveKit, Deepgram, ElevenLabs, Silero; voice side-car |
| **15** | **Voice goes production-shaped + first-class UI feature** | **Voice fast path, real streaming, barge-in memory integrity, reactive bubble, sidebar split, auto-title, latency observability triangle** |
| 16 (next) | Deploy to AWS | ECS Fargate, ElastiCache, SQS, CI/CD, Locust load test |

**Week 15 is purely additive** to Week 14 — removing `src/voice/` (the Week-15-tagged additions), `ui/src/components/Voice*.tsx`, and `src/api/routers/voice.py` leaves Week 13/14 fully functional.

---

## Dependency highlights

```
# Voice stack (Week 14)
livekit>=1.0.0
livekit-agents>=1.5.0
livekit-plugins-deepgram>=1.5.0
livekit-plugins-elevenlabs>=1.5.0
livekit-plugins-silero>=1.5.0
deepgram-sdk>=6.0.0
elevenlabs>=2.0.0
sounddevice>=0.5.0
onnxruntime>=1.17.0
torch>=2.0.0

# Week 15 — UI voice integration
livekit-client@^2.5.0           # (in ui/package.json)
framer-motion@^11.11.17         # (already present, used for bubble animation)

# MCP (Week 12)
mcp>=1.27.0
fastmcp>=3.0.0
langchain-mcp-adapters>=0.2.2
```

`pyproject.toml` and `requirements.txt` ship in lock-step. To add a dependency: edit both, or regenerate via `pip-compile pyproject.toml -o requirements.txt`.

---

**License:** MIT — for educational use within the AI Engineer Essentials course.
