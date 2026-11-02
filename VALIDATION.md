# Local engineering validation

Only pretrained Qwen3-0.6B on CPU float32 was executed locally. Qwen3-14B, CUDA BF16 and GH200 execution remain untested. No cluster job was submitted. The Python source here is byte-identical to the validated local source; its SHA256 is recorded in `validation/validation_report.json`.

A reproducible equivalent of the local smoke command (replace the checkpoint path):

```sh
python experiment.py --model Qwen/Qwen3-0.6B \
  --local-model /path/to/Qwen3-0.6B \
  --device cpu --dtype float32 --seeds 901 --chains 4 --queries 1 \
  --max-new-tokens 256 --thinking on --export-handoff --no-rebase-ablation \
  --out runs/local-smoke
python audit_results.py --run runs/local-smoke
python verify_handoff.py --run runs/local-smoke
```

The local checkpoint was the Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`. All 10 generations reached EOS. Full-text and both single-hop answers were correct; both rebased merge orders and the no-rebase ablation were wrong on this single smoke query. Missing-evidence and wrong-evidence controls were incorrect. This is engineering validation, not a new held-out accuracy study.

Checks passed:

- Eleven unit tests, including actual Qwen RoPE agreement, BF16 rotation arithmetic, zero/inverse shifts, prefix handling, cache immutability, serialization, explicit EOS/cap flags, data independence, worked-example isolation, hidden-reasoning exclusion and journal recovery.
- Ruff lint, Black formatting, syntax compilation and formal JSON Schema validation.
- Pretrained-model cache save/reload and fresh-process next logits matched exactly (maximum absolute error 0).
- Completed resume skipped model loading/generation. Partial resume regenerated exactly one missing record, retaining earlier records and producing identical regenerated token IDs.
- Changed token budget was rejected without modifying the original journal.

`validation/public_smoke.json` retains synthetic fact mappings and every generated answer/token sequence from the smoke run, plus its configuration, termination flags, timings and check results. It is explicitly a privacy-filtered snapshot, **not** an original schema-1.0.0 run directory or resume input. Machine-specific absolute paths and original run fingerprints were omitted. Original validated local artifacts remain unchanged privately. Logs and the original result content hash are retained as supporting evidence; that hash identifies the original private result, not this filtered snapshot.

For your own run, the normal `results.json` remains the complete portable analysis file. Run the documented audit on your own unmodified output folder.

## CUDA device-index correction

The original pretrained smoke above used the initial source, whose historical hash remains in the original validation report. The subsequent device-only fix resolves bare `cuda` to its current logical index before CUDA APIs, preserving explicit indices. Both local source copies are synchronized. Fourteen unit tests pass, including mocked default/explicit CUDA CLI startup and indexed synchronization/device-property queries; no real CUDA hardware or cluster was used. See `validation/cuda-device-fix.json`.

The reported `set_device(cuda)` failure occurred before output-directory creation, model loading or run metadata. It can be retried with the same command after pulling the fix if that output path does not already exist. Existing result files and schemas are unchanged and remain readable/auditable. The strict source-fingerprint resume guard is intentionally unchanged: runs started with older source must be resumed using their original source, or a changed experiment must use a new output directory.
