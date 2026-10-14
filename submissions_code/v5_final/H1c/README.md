# veritas - Guardian of Truth

A binary error detector for an LLM customer-service agent: it looks at the dialogue the agent has seen and the agent's new turn, then returns 0 or 1.

How the platform runs it:

    pip install .
    python scripts/predict.py --input <test.parquet|csv> --output <predictions>

The result is a parquet file carrying columns `id` (string) and `label` (int64). The pipeline that is actually executed is picked by `src/veritas/config.json`.

## Modules

| file | role |
|------|------|
| transcript.py | pulls apart the dialogue text: turns, tool invocations, environment replies, policy time |
| tool_schema.py | parses the `[AVAILABLE TOOLS]` section into tool specs |
| checks.py | rule-only turn checks: unknown tool, bad schema or enum, placeholders, discoverable tools used without unlocking, retry of a failed call, domain policies |
| analysis.py | walks every input example and runs the checks |
| probes.py | deterministic pipelines without a model (`probe_hard`, `probe_hardtext`, `probe_text`) |
| cli.py | command-line entry point: probe or model-based ranking |
| ranking.py, engine.py, chat.py | model-based ranking of turns over vLLM (bundled only in archives that also ship weights) |
| replay.py, audit.py, atoms.py | ranking strategies: the model plays the agent / reasoning-based review / narrow per-call questions |

## Probes (no model)

- `probe_hard` - label 1 iff at least one deterministic check fired.
- `probe_hardtext` - label 1 iff a check fired OR the turn had no tool calls.

Archives of that family bundle only parsing and checks; vLLM and the model itself are not needed.
