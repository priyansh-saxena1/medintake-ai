# medintake-ai — Debugging Catalogue
*A first-person account of every fix across all 40 commits, Apr 25–26 2026*

---

## The Project

I built a LangGraph-based clinical intake agent that holds a multi-turn conversation with a patient, collects their Chief Complaint, full HPI (Onset, Location, Duration, Character, Severity, Aggravating, Relieving), and a Review of Systems across three body systems, then emits a structured `ClinicalBrief` JSON at the end. It runs as a FastAPI web app with a chat UI, deployed on HuggingFace Spaces via Docker, and uses a local LLM (Ollama + `qwen2.5:0.5b`) for inference.

Everything was built and debugged in a single day — 40 commits, start to finish.

---

## Phase 1 — Foundation (commits 1–8)
### *Building the skeleton*

**Commit 1 — `189d4f7` initial commit**
HuggingFace Spaces auto-generated a `.gitattributes` and a 10-line boilerplate README on first push. That was the entire repo at this point.

**Commit 2 — `fa95b8a` Delete README.md**
Immediately deleted the HF boilerplate README so I could replace it with a real one.

**Commit 3 — `e977d87` init: project foundation**
Added the actual project structure in one shot: `.gitignore`, `pytest.ini`, `requirements.txt`, and a 265-line README that fully planned the architecture, API reference, usage, and deployment before a single line of app code existed.

**Commit 4 — `df4a61a` feat: add pydantic schemas and state definitions**
Created `app/__init__.py`, `app/schemas.py` (HPI, ClinicalBrief Pydantic models), and a stub `app/state.py`.

**Commit 5 — `6ea946a` feat: add LLM providers and graph orchestration**
Added `app/graph.py` and `app/llm.py` — 444 lines in one shot. This version had a full multi-node LangGraph pipeline: `intake_node → hpi_node → ros_node → scribe_node`, with static hardcoded HPI questions, CC-to-ROS keyword mapping, a regex-based severity extractor, and a `MockLLM` for tests. It also had a `TransformersLLM` that loaded Qwen 2.5 via PyTorch directly.

**Commit 6 — `56808e1` feat: add FastAPI app and CLI entry point**
Added `app/main.py` with the FastAPI `/chat` and `/health` endpoints plus a CLI mode.

**Commit 7 — `d0b0680` chore: add Docker support for HuggingFace Spaces**
First Dockerfile. It tried to conditionally download the model at build time using `hf_hub_download` and install `llama-cpp-python` unless `MOCK_LLM=true` was passed as a build arg.

**Commit 8 — `d5fb3e9` test: add end-to-end tests**
Added `tests/__init__.py` and `tests/test_e2e.py` with 151 lines covering full flow, HPI re-prompting, ROS scoping, brief structure, and the health endpoint.

---

## Phase 2 — First Fires (commits 9–13)
### *Docker breaks on first push, prompts are bad, greetings crash the flow*

**Commit 9 — `5a79774` fix: default to no llm mode**

**Problem:** The Dockerfile had a conditional `RUN` block that called `pip install llama-cpp-python` and then `hf_hub_download` to pull a 500MB+ GGUF model unless `MOCK_LLM=true` was explicitly passed as a build argument. On HuggingFace Spaces this caused the Docker build to hang, then fail — no GPU, no disk space for a model, and the build arg wasn't being passed.

**Fix:** Ripped out the entire conditional block. Changed to `ENV MOCK_LLM=true` hardcoded. Model loading became a runtime concern only. The Dockerfile went from 22 lines to 13.

---

**Commit 10 — `058b7cd` feat: add chat interface**

Added the HTML/CSS/JS chat frontend (`app/static/index.html`, ~950 lines). Mounted it as a static route in FastAPI, wired `GET /` to serve it. Updated `requirements.txt` with `python-multipart` and static file dependencies.

---

**Commit 11 — `99c13fa` fix: improve LLM prompts**

**Problem:** The LLM responses were verbose and inconsistent. The graph also had a structural bug: `_ask()` was calling `get_llm()` at module import time, which would crash if the LLM wasn't ready. The existing HPI questions were generic: "When did your symptoms first start?" with no reference to what the patient was actually experiencing.

**Fixes:**
- Moved `from app.llm import get_llm` inside `_ask()` so it's deferred to call time.
- Added a tight `SYSTEM_PROMPT` capping responses to 20 words, one question at a time.
- Replaced generic HPI questions with `{cc}`-templated versions: "When did `{cc}` start?", "Where do you feel `{cc}`?" etc. A `_fmt_question()` helper injects the first few words of the chief complaint naturally.
- Shortened severity, aggravating, and relieving field descriptions in the prompt.
- Major graph simplification: stripped ~80 lines of redundant state assignments.

---

**Commit 12 — `e7ee8c9` fix: add interrupts in graph**

**Problem:** If a patient started the conversation with "hello" or "hi" or a short affirmation like "ok", the graph stored that as their chief complaint and immediately advanced to HPI. So the first question would be "When did *hi* start?" — clearly wrong.

**Fix:** Added a `GREETINGS` set: `{"hello", "hi", "hey", "start", "begin", "ok", "okay", "yes", "sure"}`. In `intake_node`, if the user's first message is in this set or is ≤4 characters, the graph replies with a proper opening question and stays in the `intake` stage rather than advancing. The user's message is also now pulled from a slice of the message list by index rather than just taking the last message unconditionally.

---

**Commit 13 — `808ef75` fix: improve responses flow**

**Problem 1:** The ROS system selection had a bug in `get_relevant_ros_systems()` — it returned the first matching keyword's system list and exited, meaning if a complaint had multiple relevant keywords only one would be considered. Systems could also appear as duplicates.

**Problem 2:** ROS questions were scattered across the code without a canonical lookup dict.

**Problem 3:** The frontend brief panel was rendering HPI fields via `Object.entries()` on a plain object, which doesn't guarantee insertion order — fields could display in random order.

**Fixes:**
- Rewrote `get_relevant_ros_systems()` to iterate all keywords, deduplicate with a `seen` list, and return the accumulated unique list (or `DEFAULT_ROS` if nothing matched).
- Added `ROS_SYSTEM_QUESTIONS` dict with standardised clinical questions for cardiac, respiratory, GI, neuro, ENT, vision, and constitutional.
- Added `_parse_ros_answer()` to split free-text ROS answers on commas, semicolons, and "and".
- Improved severity regex to also match bare digit strings 1–10 as fallback.
- Extended `_is_vague_answer()` with "not really" and deduplicated entries.
- Frontend: changed `hpiLabels` from an object to an ordered array of `[key, label]` pairs so HPI fields always display in OPQRST order. Added Copy and Print buttons to the brief panel header. Added italic grey styling for "Not specified" values so missing fields are visually obvious.

---

## Phase 3 — Architecture Rethink (commits 14–15)
### *Seven nodes collapsed to two; state becomes a single JSON blob*

**Commit 14 — `284dfa9` feat: add dual agent architecture**

This was the largest single refactor of the project. The original graph had **seven nodes**: `intake_node`, `hpi_node`, `ros_node`, `scribe_node`, `triage_node`, `extractor_node`, `evaluator_node`, and `conversationalist_node`. Each maintained its own slice of a fragmented `IntakeState` TypedDict with fields like `chief_complaint`, `hpi` (a dict), `ros_systems`, `ros_current_index`, `ros_pending_system`, `last_processed_message_index`, and `vague_retry_field`.

**Problem:** The nodes couldn't cleanly share context. The conversationalist had to receive the current state via a separate evaluator pass. The extractor ran LLM calls just to populate a schema, and the conversationalist ran a second LLM call to generate the question. State updates were scattered across five different return dicts. Tests were constantly breaking because state fields weren't initialised consistently.

**Fix:** Collapsed everything to **two nodes**: `triage_node` (fast keyword emergency check, no LLM) and `agent_node` (one combined LLM call that does extraction AND reply generation). State shrank to a single `clinical_state: str` field holding a serialised `CombinedOutput` JSON object — all fields flat on the model. `IntakeState` went from 9 typed fields to 4. `evaluate_missing()` was replaced inline. Tests and schemas all updated accordingly.

---

**Commit 15 — `0bcdd07` fix: optimize loading**

**Problem:** The dual-agent commit introduced `ClinicalStateExtraction` as the state model with a nested `hpi: HPI` sub-object. This meant field access was `state.hpi.onset` rather than `state.onset` — two levels of nesting that made the LLM's JSON output harder to parse and the graph code more verbose.

**Fix:** Replaced `ClinicalStateExtraction` with `CombinedOutput` — a flat Pydantic model where all HPI fields live at the top level. Added `compute_stage()` to derive `intake → hpi → ros → done` purely from which fields are populated, and `missing_from()` to get an ordered list of what's still needed. Added `EMERGENCY_PHRASES` list to `triage_node` (replacing the inline hardcoded strings). Renamed `extractor_node` to `agent_node`. Format transcript helper cleaned up. Tests expanded by ~80 lines.

---

## Phase 4 — Inference Engine Hell (commits 16–21)
### *25s per turn, wrong API endpoint, wrong response field, scribe node removed*

**Commit 16 — `03af64f` chore: add more logging**

**Problem:** Inference was taking a long time and I had no visibility into where the time was going — was it model loading, tokenisation, or generation?

**Fix:** Added `time.time()` checkpoints inside `TransformersLLM._infer()` tracking tokenisation, inference, and decode phases separately. Added full request/response timing in the API. Reduced `max_tokens` from 350 to 200. Printed load time on startup.

---

**Commit 17 — `0b46033` fix: emergency issue**

**Problem 1:** The `emergency` field in `CombinedOutput` was being triggered by the LLM for any mention of "chest pain" — the prompt just said "set emergency=true for serious symptoms" which was far too broad. Any patient mentioning chest pain got a `🚨 EMERGENCY` response and the session terminated immediately.

**Problem 2:** API errors were swallowed silently, and there was no per-request log of what message was received.

**Fixes:**
- Rewrote the emergency rule in the system prompt to be much more specific: only fire on EXACT acute phrases like "crushing chest pain" or "I can't breathe right now", never on generic symptom mentions.
- Added timestamp-prefixed `[API]` log lines showing the session ID, message content, graph invoke start/end, total time, and reply length.

---

**Commit 18 — `4e16e37` feat: migrate inference engine to Ollama for 10x faster CPU inference**

**Problem:** Raw PyTorch inference via `TransformersLLM` was taking ~25 seconds per turn on CPU — completely unusable in production. Even with greedy decoding and `float16`, the model forward pass on a CPU-only machine was too slow.

**Fix:** Completely replaced `TransformersLLM` with `OllamaLLM`. Instead of loading a PyTorch model in-process, I now POST to a local Ollama HTTP server (`localhost:11434`) that runs the same model via llama.cpp under the hood — C++-optimised, ~2s per turn on CPU.

Added `startup.sh`: starts the Ollama daemon, waits for it to be healthy, pulls `qwen2.5:0.5b`, then starts FastAPI. Updated Dockerfile: install Ollama via its official install script, add `bash` and `zstd` to apt deps (required by the installer), and use `./startup.sh` as the container entrypoint. Updated `requirements.txt` to remove PyTorch/Transformers, update README to document the new architecture.

---

**Commit 19 — `c6b9370` fix: docker issue**

**Problem:** After the Ollama migration two things broke:
1. Docker build was failing because `bash` and `zstd` weren't in the `apt-get install` list.
2. The `emergency` handling inside `agent_node` was still present and was causing the node to return early before the brief was built — the emergency check was conflicting with the normal flow.

**Fix:** Added `bash` and `zstd` to the Dockerfile's apt dependencies. Removed the entire emergency block from inside `agent_node` (the `triage_node` already handles emergencies upstream, so having a second one inside the agent was redundant and dangerous). Also removed the stale emergency prompt instruction and `emergency: bool` field from the LLM output schema.

---

**Commit 20 — `2ea503f` fix: use /api/chat endpoint**

**Problem:** I had been calling Ollama's `/api/generate` endpoint. This endpoint accepts a `prompt` string — it doesn't support a `messages` array with role-based formatting. My system prompt was being prepended as a plain string `"System: ...\nUser: ..."`, which meant the model wasn't receiving it through its chat template and was largely ignoring the instructions.

**Fix:** Switched to `/api/chat` which accepts `{"messages": [{"role": "system", ...}, {"role": "user", ...}]}`. This is how the model was fine-tuned and how system instructions are actually applied. Removed the `full_prompt = f"System: {COMBINED_SYSTEM_PROMPT}\nUser: {prompt}"` string concatenation.

---

**Commit 21 — `eb1b955` fix: added new code**

**Problem 1:** After switching to `/api/chat`, all LLM responses were empty. The response parser was reading `data["response"]` — the field returned by `/api/generate`. The `/api/chat` endpoint returns `data["message"]["content"]`.

**Problem 2:** There was still a separate `scribe_node` that ran after the agent to build the `ClinicalBrief`. This added a full extra LangGraph node invocation (and potential LLM call) just to assemble a data structure that could be done inline.

**Fixes:**
- Changed response field: `raw = data.get("response", "")` → `raw = data.get("message", {}).get("content", "").strip()`
- Removed `scribe_node` entirely. Brief construction now happens inline inside `agent_node` when `compute_stage()` returns `"done"` — same turn, no extra node, no extra LLM call.

---

## Phase 5 — The Infinite Loop Crisis (commits 22–24)
### *LLM keeps asking the same question, negative answers silently dropped*

This was the most frustrating bug cluster. The agent would reach `severity` or `aggravating` and repeat the exact same question every turn regardless of what the patient answered, locking the conversation in an infinite loop.

**Root cause:** The LLM was receiving a patient answer like "it's very mild, almost nothing" or "none, zero", deciding it wasn't a "valid" answer, and leaving the field null — then re-generating the same question. The system prompt also didn't explicitly tell the model to accept negative or vague answers, so it kept probing for a more complete response that never came.

**Commit 22 — `33531b2` fix: three-layer defense against infinite loop**

**Fix 1 — Repeat detection:** Added `_detect_repeat()` that checks if the last two assistant replies in the message list are identical. Called immediately after each LLM response.

**Fix 2 — Force-fill on repeat:** When a repeat is detected and the LLM is stuck on an HPI field, force-fill that field with `"not specified"` to break the deadlock. This is the last-resort escape hatch.

**Fix 3 — Prompt: accept negative answers:** Added to the system prompt:
> "If the patient gives ANY answer (even 'none', 'zero', 'not sure', 'it goes away'), that IS a valid value. Store it as a string. For relieving/aggravating: if the patient implies rest helps, set relieving='rest'. Do NOT ask the same question twice."
> "CRITICAL: If the patient replies with 'none', 'zero', 'no', or 'nothing', you MUST extract that exact word. Do NOT leave it null."

Also added a post-parse coercion step: if the LLM returned empty string or literal `"null"` as a value, coerce it back to Python `None` so the field stays properly empty rather than storing garbage.

---

**Commit 23 — `9c76f2a` fix: field descriptions + enhanced loop guard**

**Problem:** After force-fill, the reply was silent — the user just saw nothing happen, or got "Could you tell me more?" — which looked like another loop. Also the LLM's JSON schema showed bare field names ("onset": "...") with no hint of what format was expected, making extraction unreliable.

**Fixes:**
- After force-fill, explicitly recompute `missing_from(result)` and generate a proper next question: "Thank you. Now, could you tell me about onset?" — so the user sees the conversation advance.
- Added semantic descriptions to every field in the JSON schema: `"onset": "when the symptom started"`, `"character": "quality of pain: sharp, dull, tightening, pressure, burning, squeezing, etc."`, `"severity": "how bad, e.g. mild, moderate, severe, or a number out of 10"`.

---

**Commit 24 — `cb8adc6` fix: ROS guidance + done-state guard in API**

**Problem 1:** The LLM had no specific guidance on how to conduct ROS — it would ask vague questions or sometimes ask psychological/emotional questions rather than sticking to physical symptoms.

**Problem 2:** The API was re-invoking the graph on every request, even for sessions where `frontend_stage == "done"`. This would resume a completed conversation graph and occasionally overwrite or corrupt the brief.

**Fixes:**
- Added explicit ROS instructions to the prompt: "Once all HPI fields are filled, ask about these 3 systems ONE AT A TIME: 1. Cardiac 2. Respiratory 3. GI. For each system the patient denies, store as `['no palpitations', 'no leg swelling']`."
- Added done-state guard in the API: reads `snapshot.values.get("frontend_stage")` before invoking the graph. If it's `"done"`, return the cached last reply and brief immediately without touching the graph.
- Updated the ROS example in the JSON schema from `"ros": {"system_name": ["finding1"]}` to the concrete `"ros": {"cardiac": ["findings"], "respiratory": ["findings"], "gi": ["findings"]}`.

---

## Phase 6 — ROS Is Completely Broken (commits 25–31)
### *Hallucinating systems, wrong questions, silent API crashes, frontend stalled*

**Commit 25 — `11a7703` fix: debug logging + real error messages + ROS guidance**

**Problem 1:** The API was crashing silently inside the `graph.invoke()` call. The `except` block was either absent or swallowing exceptions and returning a generic 500, which the frontend showed as a connection error with no useful information.

**Problem 2:** The snapshot check `snapshot.values.get(...)` was calling `.get()` directly on `snapshot.values` without first checking if `snapshot` or `snapshot.values` was `None` — causing an `AttributeError` on fresh sessions that had no checkpoint yet.

**Problem 3:** `snapshot.next` was being checked without a null guard.

**Fixes:**
- Wrapped the entire `/chat` handler body in `try/except Exception` with `traceback.format_exc()` — now every exception prints the full stack trace to stdout and returns a human-readable `"Server error: ExceptionType: message"` to the frontend instead of a silent failure.
- Added null guards: `has_state = bool(snapshot and snapshot.values)`, `has_next = bool(snapshot.next) if snapshot else False`.
- Added per-request debug log lines: snapshot state, stage, graph invoke timing, reply content.
- Massive frontend overhaul (~900 line diff): better error display, improved loading states, more robust message rendering.

---

**Commit 26 — `c9ecd03` fix: ROS hallucination guard + debug logging**

**Problem:** When the conversation reached the ROS stage, the LLM would fill all three ROS systems (`cardiac`, `respiratory`, `gi`) simultaneously in a single response — essentially making up all the answers in one shot without asking the patient anything. The patient never got asked about their symptoms; the LLM just invented them.

**Fix — ROS Hallucination Guard:** After each LLM call, compare the keys in `result.ros` to the keys in the previous checkpoint's `ros`. If more than one new system key appeared in a single turn, discard all but the first new one and force-set `result.reply` to the hardcoded question for the next missing system from `ROS_QUESTIONS`.

---

**Commit 27 — `f538014` fix: ROS hallucination guard + debug logging (notebook)**

Added `clinical_ai_agent_fixed.ipynb` — a Jupyter notebook version of the fixed agent, capturing the same logic in an interactive format (1132 lines added). Named `_fixed` to distinguish it from an earlier broken iteration.

---

**Commit 28 — `84c39c6` fix: force hardcoded ROS questions + update test**

**Problem:** Even with the hallucination guard, the LLM still generated vague or off-target ROS questions ("Do you have any other symptoms?") instead of asking about specific systems.

**Fix:** Extracted `ROS_QUESTIONS` as a module-level dict. Added active ROS forcing to `agent_node`: if `stage == "ros"`, bypass whatever reply the LLM generated and inject the hardcoded question for the next uncovered system directly. Fixed a subtle bug in the prev_ros loading: `prev_state.get("ros", {})` could return `None` (if the LLM had explicitly set `"ros": null`), changed to `prev_state.get("ros") or {}`.

---

**Commit 29 — `b7c799b` fix: frontend retry + ROS forcing + test update**

**Problem:** If the API returned an error mid-conversation, the JS frontend stopped completely and just displayed "Error connecting to server" — the patient had no way to recover or retry without refreshing the page.

**Fix:** Added automatic retry logic in the frontend JS. On error, the UI shows a retry button and retries the last message automatically after a short delay.

---

**Commit 30 — `27b1ed4` feat: stage-specific prompts + contextual ROS**

**Problem:** A single combined system prompt was too much cognitive load for the 0.5B parameter model. The model was constantly trying to handle HPI extraction, question generation, ROS classification, and done-detection all at once — and failing at all of them somewhat.

**Fix:** Split into three focused prompts: `INTAKE_PROMPT`, `HPI_PROMPT`, and `ROS_PROMPT`. Each contains only the instructions relevant to its stage. `combined_call()` now accepts `stage=` and selects the prompt accordingly. The graph computes `current_stage` before the LLM call using `compute_stage(pre_state)` and passes it in. Removed the graph-level hardcoded ROS question forcing since the stage-specific prompts should handle it.

---

**Commit 31 — `7f00c10` refactor: remove regex extraction, pure LLM-driven**

**Problem:** `MockLLM.combined_call()` had a ~60-line regex extraction block that tried to parse field values from the transcript string (matching "yesterday", "chest", "constant", etc.). This was fragile and was masking LLM extraction bugs in tests — the mock would extract fine while the real LLM would fail, giving false confidence.

**Fix:** Deleted the entire regex block. `MockLLM` now just walks through fields sequentially: on each call, store the patient's last message into the next unfilled field, then generate the appropriate next question. It's now a simple field-walker that makes tests predictable and ensures bugs in the real LLM can actually surface.

---

## Phase 7 — Persistent State Corruption (commits 32–40)
### *Fields vanishing, ROS history resetting, Pydantic blowing up, placeholders leaking*

**Commit 32 — `f42a7a8` fix: extend loop guard to cover ROS-stage repeats**

**Problem:** The loop guard (`_detect_repeat`) only handled HPI-stuck cases (force-filling a null HPI field). During ROS, if the LLM repeated the same question, the guard did nothing — it would check `hpi_filled`, find that HPI was complete, and fall through without any action.

**Fix:** Extended the guard with a two-branch structure:
- **HPI stuck** → force-fill the first empty HPI field with `"not specified"`, generate the next question.
- **ROS stuck** → grab the patient's last message as their answer, store it as a `patient_reported_N` placeholder in `ros`, generate a transition message.

---

**Commit 33 — `daf4268` feat: unified prompt with state visibility**

**Problem:** `HPI_FIELDS` and `ROS_REQUIRED` were defined separately in both `graph.py` and `llm.py`, causing drift.

**Fix:** Moved both to `llm.py` as the single source of truth, imported into `graph.py`. Replaced the three stage-specific prompts (`INTAKE_PROMPT`, `HPI_PROMPT`, `ROS_PROMPT`) with a single `COMBINED_SYSTEM_PROMPT` that includes a `build_state_context()` injected status block — the LLM now sees exactly which fields are ✅ filled and ❌ missing every turn.

---

**Commit 34 — `58bc4b4` fix: add new**

**Problem 1 — ROS accumulation was broken:** `result.ros` from the LLM only contained systems mentioned in the current turn. I had been assigning `result.ros` directly to the state, which meant all previously collected ROS systems disappeared on every turn. The `ros` dict was always just the most recent single turn's output.

**Problem 2 — LLM returned lists instead of strings for HPI fields:** Sometimes the model would return `"severity": ["6", "out of 10"]` instead of `"severity": "6/10"` — an array where a string was expected.

**Problem 3 — LLM re-asked a system already in the "already covered" list:** The prompt listed covered systems but the model would sometimes still ask about them.

**Fixes:**
- **ROS fix:** Use `prev_ros` (loaded from the checkpoint) as the base for every turn. Only the first new ROS key from each turn is added on top.
- **List coercion:** Added `isinstance(v, list)` check before the existing `"null"` coercion — if a field value is a list, join it as a comma string.
- **FIX 1b — Field preservation:** After each LLM call, iterate `[chief_complaint] + HPI_FIELDS`. For every field where `result` returned `None` but the previous state had a value, restore the old value. The LLM can only fill fields, never erase them.
- **Explicit "already covered" hint in state context:** Added "ℹ️ Already covered: cardiac, respiratory — DO NOT ask about these again" to the ROS phase of the state context block.

---

**Commit 35 — `d4f689b` feat: unified prompt with state visibility**

**Problem:** The `prev_ros` extraction was happening in two separate places in `agent_node` — once before the LoopGuard and once before the ROSGuard — and they were using different methods (`json.loads(current_json).get("ros")` vs `CombinedOutput.model_validate_json(current_json).ros`), which could diverge if parsing failed differently.

**Fix:** Extracted `prev_ros` (and added `prev_cs` as the full previous `CombinedOutput`) to a single try/except block at the top of `agent_node`, before any guard runs. Both LoopGuard and ROSGuard now read from the same `prev_ros` variable. The ROSGuard was updated to build on `prev_ros` when keeping allowed systems.

---

**Commit 36 — `28798fd` feat: unified prompt with state visibility**

**Problem:** The LoopGuard was treating all repeat replies identically (force-fill), but there were actually two distinct cases:
1. LLM extracted new data correctly but generated a stale/repeated reply → should only fix the reply, not force-fill any field.
2. LLM extracted nothing AND repeated the reply → truly stuck, should force-fill.

Treating case 1 as case 2 was corrupting state by adding `"not specified"` values to fields the patient had actually answered.

**Fix:** Added `_count_hpi_filled()` to count filled fields before and after the LLM call. If `curr_hpi_count > prev_hpi_count` (new data was extracted), the guard does a reply-only fix using `_HPI_NEXT_Q` to advance to the correct next question. Only if no new data was extracted does it fall through to force-fill. Added `_HPI_NEXT_Q` dict with natural-language questions for each field.

---

**Commit 37 — `f9cad5a` feat: unified prompt with state visibility**

**Multiple fixes in one commit:**

**Fix 1 — Chain-of-thought prompting:** Added `_reasoning` field to `CombinedOutput` and restructured the system prompt to require the model to fill `"_reasoning"` first (quoting the patient's message verbatim, listing every extracted fact, identifying missing fields, then choosing the next question) before writing any other field. This "think before you fill" pattern significantly improved extraction accuracy on the 0.5B model.

**Fix 2 — `_ROS_SKIP_WORDS` frozenset:** Added a set of bare non-answers that should never be stored as ROS findings: `"next"`, `"skip"`, `"ok"`, `"okay"`, `"yes"`, `"no"`, `"sure"`, `"continue"`, `"move on"`, `"nope"`, `"yep"`. These were occasionally making their way into `result.ros` when the patient said something like "yes" in response to a ROS question.

**Fix 3 — Pydantic alias fix (first attempt):** Added `_reasoning: str = Field(default="", alias="_reasoning")` and `model_config = {"populate_by_name": True}` to `CombinedOutput`. This was an initial attempt at the Pydantic alias fix that still had issues (see commit 38).

**Fix 4 — `generate_brief_narrative()` — third LLM call for proper narrative:** Added a method that makes a dedicated LLM call with `BRIEF_SYSTEM_PROMPT` to generate a professional clinical narrative string instead of just assembling patient's raw words. MockLLM also got a `generate_brief_narrative()` implementation.

---

**Commit 38 — `81151bf` fix: Pydantic alias for _reasoning field**

**Problem:** Pydantic V2 reserves names with a leading underscore for private/internal attributes. Defining `_reasoning: str = Field(...)` on a Pydantic model either got silently dropped or raised a `NameError` during validation. The `reasoning` value wasn't being stored on the model at all, so the debug logging of chain-of-thought was not working.

**Fix:** Renamed the Python attribute to `reasoning` (no underscore) but kept `alias="_reasoning"` so the LLM's JSON key `"_reasoning"` still maps to it correctly. Changed the logging from `parsed.get("_reasoning", "")` to `result.reasoning` so it uses the model attribute instead of the raw dict.

---

**Commit 39 — `a18c0eb` feat: unified prompt with state visibility**

**The biggest architectural change in the second half of the project.**

**Problem:** The single combined LLM call (extract + reply in one shot) had a fundamental timing problem: the model would extract data from the patient's message AND generate the next question in the same pass, but the question was based on the state *before* the extraction was applied. This created a one-turn lag — the model would correctly extract "onset = yesterday" but then still ask about onset because it hadn't reflected the extraction before generating the reply.

**Fix — Two-call architecture:**

Split `combined_call()` into two sequential LLM calls:
1. **Call 1 — Extract (`_extract()`):** `EXTRACT_SYSTEM_PROMPT` — focused entirely on extracting clinical facts from the patient's latest message only. Temperature 0, 300 tokens. Returns a delta dict of changed fields only.
2. **Call 2 — Reply (`_generate_reply()`):** `REPLY_SYSTEM_PROMPT` — focused entirely on asking the next question based on the fully updated state (after extraction). Temperature 0, 120 tokens. Returns `{"reply": "..."}`.

Added `_coerce_and_merge()` to merge the extracted delta onto the previous `CombinedOutput` state with proper type coercion (list → string), field preservation (never overwrite existing), and semantic classification (if patient says something "worsens" pain, classify as aggravating even if it was parsed as relieving). Added `_BARE_NOANSWER` set for ROS finding validation. Added `_fallback_reply()` for when Call 2 fails. Updated `state_context` generation to include explicit `NEXT: ask about 'onset'` instructions.

---

**Commit 40 — `271324b` feat: unified prompt with state visibility (final)**

**Multiple final fixes:**

**Fix 1 — Deterministic ROS selection:** Added `_ROS_BY_CC` dict mapping body-part keywords to relevant system lists (e.g., "knee" → musculoskeletal, neurological, vascular; "chest" → cardiovascular, respiratory, gastrointestinal). Added `_next_ros_system()` function that deterministically picks the next ROS system: first checks `_ROS_BY_CC` for CC-relevant systems, then falls back to `_ROS_GENERIC_ORDER`. Ignores `patient_reported_N` placeholder keys when counting covered systems. Exported `_next_ros_system` and `_ROS_QUESTIONS` for use in `graph.py`.

**Fix 2 — Expanded `_ROS_QUESTIONS` dict:** Full clinical questions for musculoskeletal, neurological, vascular, cardiovascular, respiratory, gastrointestinal, ENT, urinary, and integumentary systems — replacing the earlier 3-system hardcoded dict.

**Fix 3 — `placeholder_reported_N` pollution everywhere:** `compute_stage()`, `missing_from()`, and `build_state_context()` were all counting `patient_reported_N` placeholder keys (created by the LoopGuard ROS escape) as real ROS systems. This made the graph think ROS was complete when it wasn't. All three now filter with `{k: v for k, v in ros.items() if not k.startswith("patient_reported_")}` before counting. Added `_clean_ros_for_brief()` to strip these from the final `ClinicalBrief`.

**Fix 4 — Non-clinical message extraction:** Updated `EXTRACT_SYSTEM_PROMPT` to explicitly handle non-clinical messages: if the patient says "stop asking", "I already said", "hello", "ok go ahead", or expresses frustration, `_extract()` returns `{}` — no fabricated clinical data.

**Fix 5 — Reply prompt now includes suggested question:** Changed `REPLY_SYSTEM_PROMPT` and the `build_state_context()` output to include a `SUGGESTED QUESTION:` line with the hardcoded canonical question for the next field. The model is told to use it as a template and rephrase naturally if needed — guiding output format while allowing slight variation.

**Fix 6 — Severity field format:** Fixed the `generate_brief_narrative()` brief assembly to not append `/10` unconditionally (some patients gave "moderate" without a number, and "moderate/10" looks wrong).

**Fix 7 — `ClinicalStateExtraction` import removed:** Cleaned up the `graph.py` import to only pull what's actually used from `llm.py`.

---

## Summary Table — All 40 Commits

| # | SHA | Message | Category | Core Fix |
|---|-----|---------|----------|----------|
| 1 | 189d4f7 | initial commit | setup | HF Spaces init |
| 2 | fa95b8a | Delete README.md | setup | Remove HF boilerplate |
| 3 | e977d87 | init: project foundation | setup | gitignore, README, requirements |
| 4 | df4a61a | add pydantic schemas | feat | HPI + ClinicalBrief models |
| 5 | 6ea946a | add LLM providers and graph | feat | Full 7-node graph + MockLLM + TransformersLLM |
| 6 | 56808e1 | add FastAPI app | feat | /chat, /health, CLI |
| 7 | d0b0680 | add Docker support | feat | First Dockerfile |
| 8 | d5fb3e9 | add end-to-end tests | feat | 151-line test suite |
| 9 | 5a79774 | default to no llm mode | **fix** | Docker built hung on model download — hardcoded MOCK_LLM=true |
| 10 | 058b7cd | add chat interface | feat | HTML frontend + static mount |
| 11 | 99c13fa | improve llm prompts | **fix** | Tight system prompt, {cc}-injected questions, deferred get_llm() |
| 12 | e7ee8c9 | add interrupts in graph | **fix** | "hello" stored as chief complaint — GREETINGS guard added |
| 13 | 808ef75 | improve responses flow | **fix** | ROS dedup, {cc} questions, ordered HPI display, Copy/Print buttons |
| 14 | 284dfa9 | dual agent architecture | **refactor** | 7 nodes → 2; flat CombinedOutput JSON state |
| 15 | 0bcdd07 | optimize loading | **fix** | Flat fields, compute_stage(), missing_from(), renamed to agent_node |
| 16 | 03af64f | add more logging | chore | Per-phase inference timing, max_tokens 350→200 |
| 17 | 0b46033 | emergency issue | **fix** | Emergency too broad; detailed API timing logs |
| 18 | 4e16e37 | migrate to Ollama | **feat** | TransformersLLM → OllamaLLM; startup.sh; 10x faster |
| 19 | c6b9370 | docker issue | **fix** | Missing apt deps (bash, zstd); removed duplicate emergency node |
| 20 | 2ea503f | use /api/chat endpoint | **fix** | /api/generate ignores system prompt → /api/chat with messages array |
| 21 | eb1b955 | added new code | **fix** | Wrong response field (response → message.content); remove scribe_node |
| 22 | 33531b2 | three-layer defense against infinite loop | **fix** | _detect_repeat() + force-fill + accept negative answers |
| 23 | 9c76f2a | field descriptions + enhanced loop guard | **fix** | Force-fill generates new question; semantic field descriptions in schema |
| 24 | cb8adc6 | ROS guidance + done-state guard | **fix** | Explicit ROS instructions; guard against re-invoking done session |
| 25 | 11a7703 | debug logging + real error messages | **fix** | try/except with full traceback; null guard on snapshot; frontend overhaul |
| 26 | c9ecd03 | ROS hallucination guard | **fix** | LLM filling all 3 ROS in one turn → keep only first new system |
| 27 | f538014 | ROS hallucination guard (notebook) | feat | Added clinical_ai_agent_fixed.ipynb |
| 28 | 84c39c6 | force hardcoded ROS questions | **fix** | Active ROS forcing; ros or {} null guard |
| 29 | b7c799b | frontend retry + ROS forcing | **fix** | Frontend auto-retry on error |
| 30 | 27b1ed4 | stage-specific prompts + contextual ROS | **fix** | INTAKE/HPI/ROS prompts; removed hardcoded ROS forcing |
| 31 | 7f00c10 | remove regex extraction, pure LLM-driven | **refactor** | MockLLM sequential walker; removed 60-line regex block |
| 32 | f42a7a8 | extend loop guard to cover ROS-stage repeats | **fix** | Two-branch LoopGuard: HPI stuck vs ROS stuck |
| 33 | daf4268 | unified prompt with state visibility | **fix** | HPI_FIELDS/ROS_REQUIRED in one place; unified prompt + state context |
| 34 | 58bc4b4 | add new | **fix** | ROS accumulation broken; list coercion; FIX 1b field preservation |
| 35 | d4f689b | unified prompt with state visibility | **fix** | prev_ros loaded once at top; ROSGuard builds on prev_ros |
| 36 | 28798fd | unified prompt with state visibility | **fix** | Two-branch LoopGuard: reply-only fix vs force-fill; _count_hpi_filled() |
| 37 | f9cad5a | unified prompt with state visibility | **fix** | Chain-of-thought _reasoning; _ROS_SKIP_WORDS; Pydantic alias (attempt); brief narrative |
| 38 | 81151bf | unified prompt with state visibility | **fix** | Pydantic _reasoning alias properly fixed; reasoning logged correctly |
| 39 | a18c0eb | unified prompt with state visibility | **fix** | Two-call architecture (extract then reply); _coerce_and_merge(); _fallback_reply |
| 40 | 271324b | unified prompt with state visibility | **fix** | _ROS_BY_CC mapping; _next_ros_system(); placeholder cleanup everywhere; suggested questions |

---

*40 commits. One Sunday. All on a 0.5B model running on CPU.*