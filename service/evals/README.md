# Evaluation Service

Automated tooling to build ground-truth answers, replay agent runs, and score outputs with a rubric-style judge.

## Directory Map
- `src/evals/create_ground_truth.py` – runs the production YouTube agent across a curated CSV of prompts to generate golden answers.
- `src/evals/run_eval_agent.py` – feeds historical agent responses + logs into a judging agent that returns pass/fail for each checklist item.
- `src/evals/create_eval_agent.py` – defines the PydanticAI judge agent(`EvaluationChecklist`) and the checklist text.
- `src/evals/utils.py` – helper utilities (async map with progress, log simplifier, result serialization).

## Data Expectations
- Ground-truth prompts live in `data/ground_truth/*.csv`. Example columns:
  - `user_question` – natural language prompt (English/Chinese).
  - `product`, `recommend`, `reason` – structured fields for reference answers.
- Serialized runs are written to `data/ground_truth/eval-run-*.bin` (pickle). Keep these files under version control if you want deterministic evals.

## Running Pipelines
### 1. Generate ground truth
```bash
make run-ground-truth
```
- Sets `PYTHONPATH` so the script can import the agent package.
- `create_ground_truth.py`:
  - Loads the CSV.
  - Runs `create_youtube_agent()` per question (async with `map_progress`).
  - Stores simplified messages, formatted answers, and metadata.

### 2. Score agent outputs
```bash
make run-ground-truth-evals
```
- Reads the `.bin` file from the previous step.
- Builds the judge via `create_eval_agent()`.
- For each row:
  - Injects the original instructions, user question, short + detailed answers, and the full message log.
  - Requests a checklist (`instructions_follow`, `answer_relevant`, `reason_citations`, etc.).
- Aggregates the boolean flags into a Pandas DataFrame and prints per-check pass rates.

## Checklist Details
`create_eval_agent.py` enumerates `CheckName` values:
- **Instruction adherence** – followed required instructions, avoided forbidden behaviors.
- **Answer quality** – relevance, clarity, completeness.
- **Reasoning groundedness** – product/topic alignment with the ground truth, citation correctness, full coverage of reference items.

Extend the rubric by editing `CHECK_DESCRIPTIONS` and the `CheckName` enum; the judge will automatically include the new items in its prompt and schema.

## Utilities & Extensibility
- `utils.map_progress` – async semaphore + tqdm to control concurrency.
- `utils.simplify_messages` – trims tool logs before storing them in ground-truth bundles; tweak filters here if you add new tools.
- `process_result` – central place to change what gets persisted (e.g., add latency metrics).

## Tips
- Always run evaluations with the same `OPENAI_API_KEY` tier/model you plan to ship; rubric behavior can change across models.
- Store multiple `.bin` snapshots (timestamped) so you can compare revisions when prompts or retrieval strategies change.
- Pair quantitative checklist scores with manual spot checks (look at the saved `messages` field) to ensure failures are actionable.
