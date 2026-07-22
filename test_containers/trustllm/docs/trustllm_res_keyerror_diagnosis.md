# TrustLLM `Evaluation failed: 'res'` — diagnosis & defensive fix

**Status:** Root-caused. Defensive fix applied in `entrypoint.py`. Upstream library
bug documented for a follow-up (see below). Tracked as AIP-3946 (sibling of
AIP-3936, test-container robustness under epic AIP-826).

**Scope note:** This is a **test-container** bug, not a config bug.

---

## Symptom

Running the `trustllm` test container against `privacy_leakage` fails with a generic:

```
Evaluation failed: 'res'
```

It reproduces reliably on **small samples** (e.g. `max_rows=5`) but passes on the
BWI environment run (`max_rows=100`), where `privacy_leakage` produced real scores
(e.g. `RtA: 0.2075`). That environment-dependent behaviour is what made it look flaky.

## Where the error surfaces

`entrypoint.py` → `_evaluate_results()` dispatches `privacy_leakage` to the upstream
evaluator:

```python
elif dataset_name.lower() == "privacy_leakage":
    evaluation_results = evaluator.leakage_eval(generated_data)
```

Any exception from `leakage_eval` is caught and flattened into
`{"error": f"Evaluation failed: {e!s}"}`, so `KeyError: 'res'` becomes the opaque
string `Evaluation failed: 'res'`.

## Root cause

The `res` field holds the **model response**, written into each record during the
**generation** phase (it is not present in the raw dataset). The raw
`privacy/privacy_leakage.json` has 400 rows with fields
`['gt-email', 'gt-name', 'prompt', 'type', 'shotType']` — **no `res`**.

The upstream `trustllm.task.privacy.PrivacyEval.leakage_eval` iterates the 8 groups
(`shotType` ∈ {zero-shot, five-shot} × `type` ∈ {A, B, C, D}) and, for each filtered
record, does:

```python
if el['res'] != None:   # <-- hard subscript; guards the VALUE, not the KEY
    ...
```

This assumes `res` **always exists**. If generation failed to write `res` for any
record that survives the filter (silently dropped API call, thread exception, or a
small-sample edge case), `el['res']` raises `KeyError: 'res'` and aborts the *entire*
dataset evaluation.

### Why small samples fail but 100 rows pass

The dataset is ordered by `shotType`/`type`. The **first 5 rows are all
`five-shot` + `type A`**. With `max_rows=5`:

- The zero-shot groups filter to empty → `res` never accessed → fine.
- The `five-shot`/`type A` group contains those 5 records → `el['res']` is accessed.
  If generation didn't populate `res` on even one of them, it raises.

With `max_rows=100` under a healthy network, generation populated `res` on every
record, so the subscript never tripped.

## Fix applied here (defensive, in our entrypoint)

We do **not** patch the vendored upstream package. Instead `_evaluate_results` now
normalizes generated data before handing it to any evaluator:

```python
@staticmethod
def _normalize_generated_data(generated_data):
    if isinstance(generated_data, list):
        for element in generated_data:
            if isinstance(element, dict) and "res" not in element:
                element["res"] = None
    return generated_data
```

Backfilling `res=None` funnels incomplete records into the upstream value-level guard
(`el['res'] != None`) instead of crashing on the key access. Behaviour on complete
data is unchanged; incomplete records are treated as "no response", which is the
correct semantics.

### Verification

- E2E: `asqi execute-tests` with a gpt-4o-mini SUT (direct OpenAI) at `max_rows=5`
  → `success: true`, `evaluation_error: null`, real score. Previously this exact
  condition crashed with `Evaluation failed: 'res'`.
- Unit: `tests/test_res_normalization.py` covers the missing-`res` crash path and the
  normalization behaviour.

## Recommended follow-up

1. **Upstream robustness fix** — `leakage_eval` (and audit other evaluators) should
   use `el.get('res')` rather than `el['res']`. File upstream / vendor patch.
2. **Better error surfacing** — replace the generic `Evaluation failed: 'res'` with a
   message naming the dataset, the missing key, and the affected record count.
3. **Generation-phase gap** — investigate why records occasionally lack `res` on small
   samples (dropped/timed-out API calls being silently swallowed). Relates to the
   success-flag work in AIP-3920.
