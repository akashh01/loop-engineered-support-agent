This is a demo project used as an example for the blog : blog link

# Loop Engineering Demo: Self-Learning Customer Support Agent

A working before/after example where discovery is genuinely autonomous:
a customer question arriving is the trigger, nobody hands the system a
task.

## What's here

- `data/seed_docs.json` — small FAQ knowledge base (shipping, returns,
  password reset), embedded into Chroma on first run
- `knowledge_store.py` — Chroma wrapper: seed, query, add learned entries
- `llm_client.py` — Ollama chat + embeddings, Claude API swap noted
- `human_sim/human_answers.json` — canned "human" responses so escalation
  is reproducible without a live person; falls back to a real `input()`
  prompt if a question isn't in the fixture
- `before/plain_rag.py` — naive RAG: retrieve top-1, answer regardless of
  relevance, no confidence check, no escalation, no learning
- `after/support_loop.py` — the loop: retrieve → confidence gate →
  (respond directly | escalate to human → validate/reframe the answer →
  embed into permanent memory, or log for review if it's too
  vague/customer-specific to reuse)
- `after/state.py` — the `SupportState` schema
- `compare.py` — runs the same unseen question twice, before and after
  the loop has learned it

## Setup

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv sync
```

This creates `.venv` and installs everything pinned in `uv.lock`. No
separate `pip install` step needed.

**With plain pip:**

```bash
pip install -r requirements.txt
```

Either way, you'll need a local [Ollama](https://ollama.com) instance:

```bash
ollama pull qwen2.5          # chat model
ollama pull nomic-embed-text # embedding model
```

To use Claude for the chat step instead, edit `llm_client.py` (see the
commented-out swap) and run `uv sync --extra claude` to pull in the
`anthropic` package. Embeddings stay local either way — Anthropic doesn't
serve an embeddings endpoint, and Chroma needs one consistent embedding
function across every insert and query.

## Running

With uv, prefix every command with `uv run` (or run `source
.venv/bin/activate` once and drop the prefix):

```bash
# before: naive RAG, no gate, no escalation, no learning
uv run python before/plain_rag.py "How long does shipping take?"

# after: full loop
uv run python after/support_loop.py "How long does shipping take?"
#unseen question 
uv run python after/support_loop.py "Can I change the shipping address after ordering?"

# side-by-side: same unseen question, before and after learning
uv run python compare.py "Can I change the shipping address after ordering?"
```

The seed FAQ deliberately doesn't cover shipping-address changes, so
that question is guaranteed to escalate on first ask. A canned human
answer for it already exists in `human_sim/human_answers.json`.

Try a fresh question not in that fixture and you'll get a real `input()`
prompt — you're the human in the loop.
