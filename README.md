# Independent KV-cache fusion with Qwen3

A direct Python experiment for **Qwen/Qwen3-14B on one CUDA GH200**. One loaded, immutable checkpoint simulates two independent workers and a receiver through separate cache states. It is not a network or multi-GPU throughput benchmark. No scheduler script, remote execution, adapter training or custom remote model code.

## 1. Environment on TIGRIS

Start from your site's supported **CUDA-enabled PyTorch for GH200 / Linux ARM64**. Use its Python environment or a virtual environment inheriting that installation. Do not install a generic x86 CUDA wheel. This repository intentionally does **not** install or upgrade PyTorch.

```sh
python -c "import platform,torch; print(platform.machine(),torch.__version__,torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('BF16:',torch.cuda.is_bf16_supported())"
# Optional isolation while retaining the site's installed torch:
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
python -m unittest -v
```

Python 3.10+; locally tested with Python 3.11.13, PyTorch 2.7.1 and Transformers 4.57.1. Pinned dependencies are in `requirements.txt`. Tokenizers needs an ARM64 wheel or the site's compatible build. CUDA BF16/GH200 and the 14B checkpoint are **not locally hardware-tested**. The official model has 14.8B parameters: BF16 weights alone are roughly 30 GB, plus caches/activations and temporary loading memory. A 96 GB GH200 is the intended target, not a measured memory guarantee. Eager attention is the conservative default; `--attention sdpa` is optional and untested on this cluster.

## 2. Smoke run, then full run

Run directly in your already allocated interactive environment. First model use downloads the public checkpoint into your normal Hugging Face cache; use the cluster's approved cache location if needed. No 14B weights are included in this repository.

```sh
python experiment.py --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --seeds 901 --chains 4 --queries 1 --max-new-tokens 512 --thinking on \
  --export-handoff --out runs/14b-smoke
python audit_results.py --run runs/14b-smoke
# Optional fresh receiver process; loads the SAME model once again after main exits:
python verify_handoff.py --run runs/14b-smoke
```

Then use distinct held-out corpus seeds:

```sh
python experiment.py --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --seed 20261102 --corpora 50 --chains 8 --queries 4 \
  --max-new-tokens 1024 --thinking on --out runs/14b-main
python audit_results.py --run runs/14b-main
```

This default is 200 paired questions × 9 conditions = **1,800 generations**. Runtime is unknown on GH200 and depends strongly on reasoning length. Check smoke truncations and full-text solvability before starting the held-out run; choose any changes using smoke seeds only. A 1,024-token cap is configurable, not a guarantee that thinking finishes. Main results retain every capped response.

To resume, repeat the **identical command** with `--resume` appended. A completed run is detected and skipped without loading model weights. A partial corpus's caches are rebuilt without query-specific information; completed record IDs are skipped. Remote `main` is pinned to the original immutable commit on resume. Changed configuration, checkpoint contents/revision, experiment source, package versions or GPU identity cause failure: choose a new output path for changed experiments. Do not edit script/source mid-run.

```sh
python experiment.py --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --seed 20261102 --corpora 50 --chains 8 --queries 4 \
  --max-new-tokens 1024 --thinking on --out runs/14b-main --resume
```

For an existing local safetensors checkpoint/tokenizer directory, add `--local-model /absolute/model/path`. It bypasses Hub access and hashes model/tokenizer/config files to identify the exact local contents; startup hashing reads the weight files once. `--revision COMMIT_OR_TAG` selects a Hub revision, resolved to its immutable SHA at runtime. Local `--revision` is only a recorded declaration; local content hashes remain authoritative. `--seeds 101,202,303` overrides `--seed`/`--corpora`. Other options are listed by `python experiment.py --help`.

`--thinking off` disables thinking for every main control. Single-hop diagnostics use the **same thinking setting by default**; override explicitly with `--single-hop-thinking off` if desired, and the difference is recorded per row. Greedy decoding is deliberate for reproducibility; it differs from Qwen's suggested sampling defaults. Deterministic algorithms are requested and TF32 is disabled, but bitwise identity across hardware/software versions is not promised. CUDA errors from deterministic kernels should be diagnosed on the smoke run, not silently bypassed in the held-out study.

## 3. Files to send back for analysis

Download the run folder, or at minimum **`results.json`**. It embeds the manifest, complete configuration, generated mappings, source records for audit, all paired outputs, token IDs, timings, termination flags and summary. No model/cache tensor files are needed for analysis.

- `manifest.json`: schema/protocol/fingerprint, resolved model and tokenizer revision, versions/hardware/CUDA/dtype, prompt and loaded-model configuration.
- `records.jsonl`: append-only, fsynced per-generation records; authoritative journal for recovery.
- `corpora.jsonl`: one source/mapping/query-selection/cache-length record per corpus.
- `results.json`: atomic consolidated portable checkpoint after every result, explicitly complete/incomplete.
- `summary.json`: per-condition normalized/strict counts, EOS/cap counts, unavailable final answers, EOS-and-correct counts, mean generation time.
- `runtime_checks.json`: first-corpus save/reload tensor and logit equivalence; temporary cache file is removed.
- `audit.json`: offline schema/invariant/scoring checks after you run `audit_results.py`.
- `load_metadata.json`, `attention_source.py.txt`, `rotary_source.py.txt`: loading time and implementation provenance.
- `last_interruption.json`: last exception/interrupt, if any. Historical evidence is retained even after successful resume; `results.json` status is authoritative.

Tensor caches are **not saved persistently by default**. `--save-caches` saves A/B tensors for every corpus and can be large. `--export-handoff` saves just the first A cache and its later query/reference logits for the optional fresh-process test (`handoff-a.pt`, `handoff-logits.pt`, `handoff.json`, then `fresh_process_check.json`). Remove these optional `.pt` files from a result archive sent for analysis if desired.

Safe interruption: Ctrl-C records an incomplete checkpoint. Power loss can leave an incomplete final JSONL line; resume preserves that tail in a `.torn-tail-*.bin` file and truncates only that torn line, never an interior corruption. Concurrent writers are rejected by a file lock. If interruption occurred between journal append and checkpoint replacement, JSONL remains authoritative and resume rebuilds the consolidated checkpoint. Use `--allow-incomplete` with the offline audit for a graceful incomplete checkpoint; it never relabels the run complete. There is no silent row deletion or test-informed best-order selection.

## Method

Each corpus has randomized one-to-one person→locker→code mappings. Names/locker labels are sampled, and the same first N color-code labels are permuted within every corpus. A and B fact orders are independently shuffled. All independent caches, wrong-evidence cache, shared-prefix cache and joint-text reference cache are built **before** the later query targets are sampled. Multiple targets reuse the same pristine source caches. Receiver generation gets only cloned tensors and later question tokens; no source-document text is replayed in a cache-only condition. The driver also constructs separate full-text controls and retains documents solely for offline audit.

Both workers encode the same system/open-user prefix independently. The receiver retains it once. Body blocks retain their original starting position P. Qwen3 stores post-RoPE keys; the merge calls the model's actual rotary module with float32 trig and applies the **incremental** position shift `target_start - original_start` to keys, casting once back to cache dtype. Values are unchanged. For A+B, B's net shift is A's document length. For B+A, B retains its original position and A shifts by B's document length. No double RoPE application, duplicated chat prefix or retrospective cross-document hidden-state computation. The script refuses non-Qwen3, scaled RoPE and sliding-window configurations rather than silently assuming compatibility. Explicit positions/cache positions and causal query masking accompany every forward call.

Conditions reported separately:

1. Full-text joint prefill (reference; not guaranteed an upper bound).
2. No evidence.
3. Only A.
4. Only B.
5. Merged A+B.
6. Merged B+A.
7. Wrong partner: B's code labels are cyclically deranged, preserving its locker labels/fact order and giving every target an incorrect code.
8. Single-hop A: person→locker retrieval.
9. Single-hop B: locker→code retrieval.

Optional `--no-rebase-ablation` adds a tenth condition, A+B with no incremental key shift. This is a diagnostic, not a CanonicalMerge reproduction. Both merge orders are always reported; do not pick the better test order after seeing results.

Thinking uses explicit Qwen3 chat/think tokens, consistent across the main controls. Hidden reasoning is never graded. If thinking is enabled and `</think>` is absent, final text is unavailable and both scores are false. Otherwise strict scoring requires the stripped final text to equal the expected answer exactly (case-sensitive). Normalized scoring accepts a unique known code label in final text, or a bare/uniquely stated locker letter for A's single-hop query; ambiguity and UNKNOWN fail. Repeated mentions of the same code are accepted. EOS completion and token-cap termination are separate fields: a capped response with a closed thinking section and a valid final answer can be normalized-correct, but remains **truncated**, never called EOS-complete. Summary reports both accuracy and `eos_and_correct` counts so this is visible. Raw output with and without special tokens is retained.

Timing uses CUDA synchronization around measured phases: each source prefill, condition cache construction, pristine-cache clone, query tokenization, question-forward/first-token, remaining decode, and total receiver operation. Prefill times are per corpus/attempt; do not sum duplicated copies attached to rows. Generation time includes the question forward and excludes merge/clone/model-load. Receiver total includes clone/tokenization/generation and final decoding. CUDA peak allocated/reserved memory includes the resident model and all live source caches; it is not incremental network-worker memory or an end-to-end distributed speedup claim.

## Validation and troubleshooting

```sh
python -m unittest -v
python audit_results.py --run runs/14b-main
```

Eleven unit tests cover actual Qwen rotary agreement, zero/inverse shifts, BF16 rotation arithmetic, prefix removal, unchanged values, fresh-cache non-aliasing, tiny randomly initialized Qwen3 generation/serialization, independent data generation, hidden-reasoning scoring, and torn-journal handling and exclusion of worked-example entities from evidence pools. `validation/` includes a privacy-filtered pretrained Qwen3-0.6B engineering snapshot and check logs; it is not a resumable run directory. They are engineering validation, not a new held-out research result. See `VALIDATION.md` for exact executed commands and limits.

If CUDA is unavailable, fix the site's PyTorch/CUDA environment; this repository does not choose a CUDA wheel. If out of memory, try the small smoke configuration first; reduce chains/token budget in a **new** run, or ask for analysis of memory metadata. Do not move a changed configuration into an existing run folder. If full-text or single-hop controls fail, merged-cache failure is inconclusive. If token caps dominate, increase the budget on smoke data before defining the held-out run. Quantized or custom/scaled-RoPE variants require separate validation and are intentionally unsupported here.

Official references: [Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B), [model configuration](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json).
