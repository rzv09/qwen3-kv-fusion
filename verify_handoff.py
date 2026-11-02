#!/usr/bin/env python3
"""Optional fresh-process, one-source handoff; reads no original source records."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import Block, Engine, atomic_json, normalize_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--local-model", help="Override location of the SAME checkpoint if moved"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    handoff = json.loads((run / "handoff.json").read_text())
    config = manifest["config"]
    device = normalize_device(config["device"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    local = args.local_model or config["local_model"]
    source = local or manifest["model"]["model_id"]
    kwargs = dict(local_files_only=bool(local), trust_remote_code=False)
    if not local:
        kwargs["revision"] = manifest["model"]["model_resolved_revision"]
    tok = AutoTokenizer.from_pretrained(source, **kwargs)
    model = (
        AutoModelForCausalLM.from_pretrained(
            source,
            dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attention"],
            **kwargs,
        )
        .to(device)
        .eval()
    )
    engine = Engine(model, tok, device)
    cache = Block(
        tuple(
            torch.load(run / handoff["cache"], map_location=device, weights_only=True)
        )
    )
    actual = engine.first_logits(cache, handoff["query"])
    expected = torch.load(
        run / handoff["reference"], map_location="cpu", weights_only=True
    )
    error = float((actual - expected).abs().max())
    report = dict(
        fresh_process_receiver=True,
        source_records_read=False,
        max_abs_logit_error=error,
        logits_exact=torch.equal(actual, expected),
        cache=handoff["cache"],
        run_fingerprint=manifest["fingerprint"],
    )
    atomic_json(run / "fresh_process_check.json", report)
    assert report["logits_exact"], f"Cache/weights/runtime mismatch: {error}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
