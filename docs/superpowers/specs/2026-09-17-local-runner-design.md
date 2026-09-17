# mobile_run_task Local Profile: On-Device Model Runner

Date: 2026-09-17
Status: Design approved, not yet implemented

## Problem

`mobile_run_task` autonomously drives a device using Flash or Pro, both of
which require a paid LLM credential (Gemini, OpenAI, Anthropic, etc.). The
companion Google AI Edge Gallery app now exposes a free, on-device model over
an OpenAI-compatible local HTTP server (`docs/superpowers/specs/2026-09-17-
local-api-server-design.md` and `...-api-server-mode-design.md` in the
`google-ai-edge-gallery` fork). The existing `mobile_observe`/`mobile_act`
"manual mode" (PR #114) already proves the device-control side of that
integration works without any Artemis LLM credential, but it requires the
calling MCP host to drive the turn-by-turn loop itself.

This feature closes the gap: a fully autonomous `mobile_run_task` run, backed
entirely by the free on-device model, needing zero paid API keys.

## Goals

- `mobile_run_task(model="Local", task_desc=...)` runs an autonomous loop
  against the Gallery's local API server, with no Artemis LLM credential
  required.
- Reuses the existing daemon dispatch, background spawn + watchdog,
  `trace_id`, `mobile_manage_task`, and `mobile_inspect_trace` machinery
  unchanged — the same operational surface Flash/Pro already have.
- Device actions and step recording reuse Flash's proven executor and
  DataEngine integration, not a new device-control stack.

## Non-goals (this iteration)

- No LangChain/`ChatOpenAI` integration for the local model. The Gallery's
  on-device model is not expected to reliably emit OpenAI-style native
  `tool_calls`; a hand-rolled JSON-in-prompt protocol is more robust for a
  small model and avoids gambling the whole feature on that behavior.
- No `TranscriptLedger` / history-chunk-compression reuse. That machinery
  exists to manage long, multi-provider LangChain message histories; a small
  on-device model with a short context window gets a much simpler bounded
  plain-text turn history instead.
- No Pro-style planning/checking/notes. This is a Flash-shaped reactive loop,
  not a Pro one.

## Decision: a third profile, not a new tool

`mobile_run_task`'s `model` parameter gains a third accepted value, `"Local"`,
alongside `"Flash"`/`"Pro"`. Internally, `artemis/sdk/agent.py`'s `run_task()`
gains a third branch next to its existing
`if request.profile.lower() == "flash": runner = FlashRunner(...)` (around
line 685):

```python
elif request.profile and request.profile.lower() == "local":
    runner = LocalRunner(context, goal=request.goal, max_turns=max_turns)
```

This was chosen over a standalone `mobile_run_task_local` MCP tool because
the entire operational stack around `mobile_run_task` — daemon dispatch
(`submit_task_to_daemon`), the standalone background-subprocess spawn +
watchdog (`mcp_server/background/task_runner.py`), `trace_store`,
`mobile_manage_task`, and `mobile_inspect_trace` — is already generic over
"whatever profile string was requested." Duplicating that stack for a new
tool would add a large amount of surface area for no behavioral benefit.

`mcp_server/tools/task_runner.py`'s model validation
(`model.lower() not in ("flash", "pro")`) extends to accept `"local"`, and
`mcp_server/background/task_runner.py`'s `task_builder.using_profile(...)`
call passes `"local"` through the same way it already special-cases
`"flash"`.

## Architecture

```
mobile_run_task(model="Local", task_desc=...)
       │
       ▼
 (existing daemon dispatch / background spawn — unchanged)
       │
       ▼
Agent.run_task()  [artemis/sdk/agent.py]
       │  profile == "local"
       ▼
LocalRunner(ctx, goal, max_turns)          [new: artemis/agents/local/runner.py]
       │
       ├─ UnifiedMobileController + McpActionExecutor   (same as FlashRunner)
       ├─ ctx.data_engine.record_step(...)              (same as FlashRunner)
       └─ turn loop: screenshot+elements → HTTP prompt → parse JSON → act
                            │
                            ▼
              Gallery local API server (/v1/chat/completions)
```

### `LocalRunner`

New file `artemis/agents/local/runner.py`. Same external shape as
`FlashRunner` (`__init__(self, ctx, goal, max_turns=None)`,
`async def run(self, state) -> dict`) so it drops into `agent.py`'s existing
per-profile branch pattern unmodified.

Reused from Flash's stack, unchanged:
- `UnifiedMobileController` for device I/O.
- `McpActionExecutor` for dispatching one validated action and getting back
  a post-action screenshot + element list — but bound to the **restricted
  action vocabulary** manual mode already uses
  (`artemis.mcp.action_manifest.REQUIRED_ACTIONS | OPTIONAL_ACTIONS`), not
  Flash's full tool set. `ask_explorer`, `video_analyzer`, and the note/
  history tools are excluded for the same reason `mobile_act` excludes them
  (PR #114): they invoke Artemis's own LLM router internally, which would
  silently reintroduce the paid-credential requirement this feature exists
  to remove.
- `capture_screenshot_and_parse_ui` for the initial and per-turn screenshot +
  UI element list.
- `ctx.data_engine.allocate_step_id()` / `record_step(...)` every turn, so
  `mobile_manage_task(action="status")`'s progress reporting and
  `mobile_inspect_trace` work against a Local run exactly as they do for
  Flash today (the DataEngine step shape is the same; no new inspection code
  needed).

Not reused: `TranscriptLedger`, chunk compression, LangChain message types,
`bind_tools`. Turn history is a plain bounded Python list of
`(turn_number, action_taken, one_line_outcome)` tuples, rendered as plain
text into the next prompt — capped (default last 8 turns) to control prompt
size for a small-context model.

### Turn loop

Each turn:

1. Capture screenshot (JPEG bytes) + numbered UI element list via the same
   helper Flash uses.
2. Render the prompt: a static instruction block (goal, the action
   vocabulary with argument shapes, and a "reply with ONLY a single JSON
   object, no prose" contract) + the bounded turn-history text + the current
   screenshot (base64, inline in the user message, same shape the Gallery
   API server already accepts — proven in
   `docs/superpowers/specs/2026-09-17-local-api-server-design.md`) + the
   numbered element list.
3. `POST {api_base}/v1/chat/completions` via plain `httpx`, bearer-token
   authenticated, `temperature: 0`. Not LangChain — a direct request/response
   against the same OpenAI-compatible shape `ChatCompletionsMapping.kt`
   already implements on the Gallery side.
4. Parse the reply's content for a JSON object matching
   `{"action": str, "args": dict, "thought": str, "done": bool, "result": str?}`.
   - Accepts a bare JSON object or one fenced in a ```json block (mirrors
     `FlashRunner._resolve_tool_calls`'s existing fallback parser shape,
     reimplemented standalone rather than imported, since Flash's version is
     entangled with LangChain tool-call objects).
   - On parse failure: append one corrective system turn ("your last reply
     was not valid JSON — respond with ONLY the JSON object") and retry once.
     A second consecutive parse failure ends the task as failed.
5. `done: true` ends the loop; `result` (if present) becomes the task's
   final report, in the same shape `report_task_status` produces for Flash
   (`{"status": "completed", ...}` / `{"status": "failed", "explanation":
   ...}`), so `mobile_manage_task`'s existing result-formatting code needs no
   changes.
6. Otherwise, dispatch `action`/`args` through `McpActionExecutor.execute`,
   record the DataEngine step, append the turn to history, and loop — until
   `done` or `max_turns` is reached (config default below).

### Config

A new, deliberately separate config block in `artemis.jsonc` — not folded
into the existing `LLMConfig`/`ModelEndpoint` types, since those are shaped
around LangChain chat model construction and this path never constructs one:

```jsonc
// artemis.jsonc
"local_model": {
  "api_base": "http://127.0.0.1:8080/v1",
  "token": "a2b11202-5536-413d-bc38-455c337cce0d",
  "model": "Gemma-4-E2B-it",
  "max_turns": 40
}
```

A new small loader, `artemis/config/local_model.py`
(`LocalModelConfig` pydantic model + `parse_local_model_config()`), mirrors
`config/llm.py`'s `parse_llm_config()` shape (same `load_jsonc` +
`get_config_path` helpers) but is a fraction of the size — no provider
dispatch, no fallback chain, no per-node overrides.

(Note: there is already an unrelated, uncommitted experimental
`gallery-local` entry under the existing `nodes` LLM config block in this
repo's working tree — that was an earlier exploration of routing Flash's
LangChain `ChatOpenAI` at the Gallery server directly, which this design
supersedes. It's left alone; the new `local_model` block is independent of
it and the plan will not depend on that entry existing.)

## Error handling

| Condition | Behavior |
|---|---|
| Gallery server unreachable / connection refused | Retry twice with short backoff; on continued failure, task fails with a clear network error — same `trace_store.update_trace_status(..., "failed", error=...)` path Flash already uses. |
| Gallery server returns non-200 (e.g. 503 "Model not initialized") | Same retry-then-fail behavior as above; the error message passes through the HTTP body's `error.message` when present. |
| Model reply is not valid JSON, twice in a row | Task fails with `"Model returned non-JSON output twice in a row."` |
| `action` names something outside the allowed vocabulary | Treated like a parse failure: one corrective retry, then fail. |
| `max_turns` reached without `done: true` | Task fails with `"Max turns reached without a final result."` — same shape as Flash's `"Max turns reached without final status report."` |

## Testing

- Unit tests for the JSON-reply parser (pure function, no device/network):
  bare JSON object, JSON fenced in ```json, JSON with leading/trailing prose,
  malformed JSON, action name outside the allowed vocabulary.
- Unit tests for the prompt builder: goal + bounded turn history render
  correctly, and the turn-history cap actually bounds prompt growth over many
  turns.
- Unit test for `LocalModelConfig` parsing (defaults, overrides, missing
  file).
- Manual on-device verification (extends the pattern from the Gallery specs'
  own manual verification sections): run
  `mobile_run_task(model="Local", task_desc="Open Settings and toggle Wi-Fi")`
  against the already-verified Gallery server, poll via
  `mobile_manage_task(action="status")`, confirm the task completes and
  `mobile_inspect_trace` shows a normal step-by-step trace.

## Related but out of scope here

- **Issue B** (some models fail to download in the locally-built Gallery app)
  — separate, Gallery-side, not addressed here.
- Making the local model usable for Pro-style planning/verification — not
  attempted; the small on-device model is not expected to handle that
  reliably, and Flash-shaped reactive execution is the right fit for now.
