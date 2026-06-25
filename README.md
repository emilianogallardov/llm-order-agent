# LLM Order Agent

A small, production-shaped agent that turns a messy natural-language purchase
order into a **structured, validated, contract-priced payload** for a food-service
procurement system.

The point of this project is one idea:

> **The model proposes. Deterministic validators own correctness.**

An LLM is great at reading "50 lbs of sharp cheddar from the main dairy co" and
proposing what it maps to. It is the wrong thing to trust with money, vendor
approval, or a final yes/no. So the LLM only ever emits *candidates*; a plain,
fully-tested Python layer resolves those candidates against the catalog, applies
contract pricing with exact decimal math, and **fails closed** on anything it
can't resolve with certainty.

## What it does

Given an order like:

```
Order 50 lbs of sharp cheddar from the main dairy co, and 20 cases of romaine
from the green produce distributor. Make sure the cheese matches our contract rate.
```

it produces:

```json
{
  "status": "ready_for_staging",
  "order_total": "865.00",
  "lines": [
    { "product_id": "PRD-CHED-SHARP-WHITE", "vendor_id": "SUP-DAIRYCORP",
      "quantity": "50", "uom": "lb",  "contract_unit_price": "4.50", "line_total": "225.00" },
    { "product_id": "PRD-ROM-CLEANED",     "vendor_id": "SUP-FRESHGREEN",
      "quantity": "20", "uom": "case","contract_unit_price": "32.00","line_total": "640.00" }
  ]
}
```

When the order is ambiguous or can't be resolved, it does **not** guess. It
returns `clarification_required` (with a drafted question) or `validation_blocked`
(with a reason), and stages nothing.

## Architecture

```
free text ─▶ model selector ─▶ LLM (proposes) ─▶ deterministic validators ─▶ payload
             (which model)      ExtractedOrder      ResolvedOrder
```

| Layer | File | Responsibility |
|-------|------|----------------|
| **Model selector** | `order_agent/models.py`, `tasks.py`, `selector.py` | Declarative model registry + task-based routing. Callers ask for a *task* (`order_extraction`); the selector resolves model, fallback, and any env override. |
| **Prompt** | `order_agent/prompts.py` | Constrains the model to the catalog, forces strict JSON, tells it to flag ambiguity and never compute prices. |
| **LLM client** | `order_agent/llm.py` | One interface, provider chosen from model metadata (OpenAI / Anthropic). `MockClient` makes the pipeline runnable offline and in tests. |
| **Validators** | `order_agent/validators.py`, `grounding.py`, `uom.py` | The decision layer: text-grounding of model facts, quantity/UOM checks, UOM canonicalization (`lbs`→`lb`), attribute-based SKU resolution, alias→vendor resolution by id, parent-entity resolution, effective-dated contract pricing, `Decimal` math, fail-closed. |
| **Catalog** | `order_agent/catalog.py`, `catalog.json` | Products, vendors (with parent hierarchy), approved aliases, contracts. Everything keyed by canonical id, never display name. |

### The model selector

Routing lives in one place so you never hard-code a model at a call site:

```python
from order_agent.selector import get_execution_config

cfg = get_execution_config("order_extraction")
cfg.model         # "gpt-4o-mini"  (or whatever MODEL_ORDER_EXTRACTION overrides to)
cfg.fallback      # "gpt-4o"
cfg.model_config  # full metadata: provider, pricing, capabilities
```

Swap the model per environment with an env var, no code change:

```bash
MODEL_ORDER_EXTRACTION=gpt-4o python run.py --live "..."
```

### Why the design choices matter

- **Every model-extracted fact is grounded against the order text.** The model
  picks the quantity, vendor reference, and attributes; the validator checks each
  one actually appears in what the buyer wrote. A model that inflates `50 → 5000`,
  swaps an unapproved vendor for an approved one, or switches `sharp → mild` is
  caught, because the new fact isn't supported by the text. The model proposes;
  the text is the source of truth.
- **The model's product pick is a hint, not the decision.** The validator
  re-derives the SKU from the (grounded) attributes and refuses to lock a line
  unless they resolve to exactly one product. A confidently-wrong model gets
  caught instead of trusted.
- **Fail closed, never crash.** Empty output, malformed JSON, a non-finite
  quantity, or a wrong-typed field all become a blocked payload, not an exception
  and not a half-built order.
- **Canonicalize units; never convert them.** `lbs`/`pounds`/`cases` collapse to
  one token (`lb`/`case`) so real phrasing resolves, but a per-lb price is never
  multiplied by a case count: different physical units stay distinct and block.
- **Display names drift; ids don't.** Vendors get renamed and reorganized.
  Contracts bind to a stable parent id, so a rename can't silently break pricing.
- **`Decimal`, never `float`.** Currency is fixed-decimal, rounded once, per line.
- **Fail closed.** No resolved SKU, no approved vendor, no active contract, or a
  UOM mismatch → block. The system never invents a price to look helpful.
- **Schema is stable across statuses.** Every response has the same keys, so a
  consumer (or a regression test) can rely on the shape.

## Run it

No dependencies needed for the demo or the tests (standard library only).

```bash
# offline demo — uses a built-in mock extraction
python run.py --demo

# the eval suite (golden-set regression tests)
python -m evals.test_cases        # or: pytest -q
```

Run live against a model:

```bash
pip install openai            # or: pip install anthropic
cp .env.example .env          # add a provider key
OPENAI_API_KEY=... python run.py --live "50 lbs sharp cheddar from the main dairy co, 20 cases romaine from the green produce distributor"
```

## Evals

The suite asserts on the deterministic payload, so it's reproducible and free
(the LLM step is mocked with a fixed extraction; the same prompts run live):

| Test | What it proves |
|------|----------------|
| `happy_path_resolves_to_865` | Alias + grounded-attribute resolution + contract pricing → exact `$865.00`. |
| `ambiguous_order_requires_clarification` | Missing variant/vendor → `clarification_required`, nothing staged. |
| `confidently_wrong_model_pick_is_caught` | Model commits a SKU on bare "cheddar" → validator clarifies anyway. |
| `attributes_override_a_wrong_model_pick` | Text says "mild shred", model guessed "sharp" → resolves to the mild SKU + `$3.80`. |
| `quantity_inflation_is_blocked` | Text "50", model emits 5000 → ungrounded quantity blocks. |
| `vendor_swap_is_blocked` | Model swaps in an approved vendor not named in the text → blocks. |
| `attribute_swap_forces_clarification` | Model's attributes contradict the text → dropped → ambiguous → clarify. |
| `unsupported_constraint_is_blocked` | Buyer asked "organic" (not in catalog) → block, don't stage a non-organic SKU. |
| `empty_order_is_blocked` | Empty model output is a failed parse, not a `$0` order. |
| `malformed_shapes_fail_closed` | Bad JSON shape / wrong-typed unit / non-finite qty → blocked, never crashes. |
| `future_dated_contract_is_ignored` | Only contracts effective on/before today are eligible; latest wins. |
| `supplier_rename_survives_via_canonical_ids` | Vendor reorg with stable parent id → still `$4.50`, byte-stable payload across 5 runs. |
| `broken_supplier_hierarchy_fails_closed` | Unresolvable parent → `validation_blocked`, no fabricated price. |
| `uom_synonyms_are_canonicalized` | `lbs`→`lb` so a correct order isn't blocked on plurals. |
| `uom_mismatch_is_blocked` | A per-lb price is never multiplied by a case count. |

```
$ python -m evals.test_cases
...
15/15 passed
```

Several of these (grounding, empty-order, malformed-shape, effective-date) were
added after an adversarial code review that ran attacks against the validator and
found that model-extracted facts weren't being checked against the order text.
Closing those is what moved this from "good prototype" to grounded and fail-closed.

### Live stress test

`stress_test.py` runs sixteen deliberately sloppy human messages ("yo can u throw
together 50 lbs of that sharp white cheddar from the main dairy co...") through
the **real model** against a catalog full of lookalike variants and competing
vendors, and runs the whole set several times to surface flakiness (a case that
passes one run and fails another is a reliability bug, not a pass). The probes
include typos ("CHEDDER"), uppercase, a missing vendor, an off-catalog product,
decimal and zero quantities, "shredded"→mild-SKU resolution, and a four-line
order. This is what caught the UOM-canonicalization gap above.

```bash
pip install anthropic
ANTHROPIC_API_KEY=... MODEL_ORDER_EXTRACTION=claude-sonnet-4-6 python stress_test.py 5
# -> 16/16 cases passed every run (80/80 calls)
```

## Limits (what grounding can't close, and the honest fix)

Deterministic grounding closes a lot: word-bounded matching (so "oil" can't match
"foil" or "premier" match "premiere"), per-line span checks, rejection of spans
that cover two line items, common vendor-negation patterns ("not premier", "do
not use premier"), unsupported-constraint detection, and catalog keyword aliases
so synonyms like "evoo" still resolve. What it still can't do alone:

- **Arbitrary negation / natural language.** Common patterns are blocked, but
  "we broke up with Premier last year, use anyone else" is beyond substring rules.
- **Omitted constraint.** If the buyer says "organic" and the model simply drops
  the attribute entirely, there's nothing in the extraction to catch.
- **A lie inside one honest-looking span.** Quantity+unit ground against a single
  per-line span, but if a model returns a verbatim one-line span and still
  misreads within it, deterministic checks can't see intent.

The real fix for the residue is the same: **constrained extraction that returns
character offsets** (so the validator checks the quantity token actually sits next
to the product token), plus a **confidence score that routes low-confidence lines
to human review**. This repo deliberately stops at the deterministic layer and
documents the seam, rather than pretending substring checks are airtight.

## Project layout

```
llm-order-agent/
├── catalog.json              # products, vendors (+ parent hierarchy), aliases, contracts
├── run.py                    # CLI: --demo (offline) | --live
├── order_agent/
│   ├── models.py             # model registry (id -> pricing/capabilities)
│   ├── tasks.py              # task -> model routing (+ env overrides)
│   ├── selector.py           # get_execution_config(task)
│   ├── prompts.py            # the extraction system prompt
│   ├── llm.py                # provider-agnostic client + MockClient
│   ├── schema.py             # ExtractedOrder (proposal) / ResolvedOrder (decision)
│   ├── catalog.py            # id-keyed catalog access + alias/parent/effective-date resolution
│   ├── grounding.py          # check model facts appear in the order text
│   ├── uom.py                # unit canonicalization (lbs -> lb), never conversion
│   └── validators.py         # deterministic resolution + grounding + Decimal pricing
└── evals/test_cases.py       # golden-set regression tests
```

## License

MIT
