#!/usr/bin/env python3
"""Offline audit of portable JSON; standard library only, no GPU/model needed."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


CODES = (
    "red blue green gold pink white black orange purple silver brown cyan lime "
    "navy teal beige coral ivory maroon olive peach violet indigo amber"
).split()
LOCKERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")


def hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized(text, thinking, kind):
    if "</think>" in text:
        final = text.rsplit("</think>", 1)[1].strip()
        if "<think>" in final:
            return None, None
    elif thinking or "<think>" in text:
        return None, None
    else:
        final = text.strip()
    if kind == "code":
        choices = set(re.findall(r"\b(" + "|".join(CODES) + r")\b", final.lower()))
    else:
        clean = final.strip(" \t\n.*`")
        if len(clean) == 1 and clean.upper() in LOCKERS:
            return clean.upper(), final
        choices = {
            x.upper() for x in re.findall(r"\blocker\s+([A-Z])\b", final, re.I)
        } & LOCKERS
    return (next(iter(choices)) if len(choices) == 1 else None), final


def audit(run, allow_incomplete=False):
    data = json.loads((run / "results.json").read_text())
    assert data["schema_version"] == "1.0.0"
    manifest, rows, corpora = data["manifest"], data["results"], data["corpora"]
    config = manifest["config"]
    calculated = hash_json(
        dict(
            schema=manifest["schema_version"],
            protocol=manifest["protocol"],
            config=config,
            source_sha256=manifest["source_sha256"],
            torch=manifest["runtime"]["torch"],
            transformers=manifest["runtime"]["transformers"],
            cuda=manifest["runtime"]["cuda_runtime"],
            packages=manifest["runtime"]["packages"],
            gpu=manifest["runtime"].get("gpu"),
        )
    )
    assert calculated == manifest["fingerprint"], "Manifest fingerprint corrupt"
    assert data["status"] in ("complete", "incomplete")
    if not allow_incomplete:
        assert (
            data["status"] == "complete"
        ), "Incomplete run; use --allow-incomplete to inspect without calling it complete"
    jsonl = [
        json.loads(line)
        for line in (run / "records.jsonl").read_text().splitlines()
        if line
    ]
    assert (
        rows == jsonl
    ), "Consolidated checkpoint lags or differs from authoritative JSONL"
    by_corpus = {c["corpus_id"]: c for c in corpora}
    assert len(by_corpus) == len(corpora)
    seen = set()
    valid_conditions = set(config["conditions"])
    for corpus in corpora:
        assert corpus["run_fingerprint"] == manifest["fingerprint"]
        assert corpus["targets_sampled_after_cache_creation"] is True
        raw_keys = (
            "seed",
            "entries",
            "order_a",
            "order_b",
            "source_a",
            "source_b",
            "wrong_b",
            "wrong_mapping",
            "target_selection_seed",
        )
        assert hash_json({k: corpus[k] for k in raw_keys}) == corpus["source_sha256"]
        assert len(corpus["entries"]) == config["chains"]
        assert len({e["person"] for e in corpus["entries"]}) == config["chains"]
        assert len({e["locker"] for e in corpus["entries"]}) == config["chains"]
        assert len({e["code"] for e in corpus["entries"]}) == config["chains"]
        assert corpus["query_entry_indices"] == random.Random(
            corpus["target_selection_seed"]
        ).sample(range(config["chains"]), config["queries"])
        assert all(
            corpus["wrong_mapping"][e["locker"]] != e["code"] for e in corpus["entries"]
        )
        assert [s.split()[1] for s in corpus["source_b"].splitlines()] == [
            s.split()[1] for s in corpus["wrong_b"].splitlines()
        ]
    for row in rows:
        assert row["schema_version"] == data["schema_version"]
        assert row["run_fingerprint"] == manifest["fingerprint"]
        assert row["record_id"] not in seen, "Duplicate row"
        seen.add(row["record_id"])
        assert row["condition"] in valid_conditions
        corpus = by_corpus[row["corpus_id"]]
        qi = int(row["query_id"][1:])
        assert (
            row["record_id"]
            == f"{row['corpus_id']}/{row['query_id']}/{row['condition']}"
        )
        assert 0 <= qi < config["queries"]
        assert row["query_entry_index"] == corpus["query_entry_indices"][qi]
        entry = corpus["entries"][row["query_entry_index"]]
        expected = (
            entry["locker"] if row["condition"] == "single_hop_a" else entry["code"]
        )
        assert row["person"] == entry["person"] and row["expected_answer"] == expected
        predicted, final = normalized(
            row["generated_text"], row["thinking"], row["answer_kind"]
        )
        assert predicted == row["parsed_answer"] and final == row["final_text"]
        assert row["normalized_correct"] == (
            predicted is not None and predicted.lower() == expected.lower()
        )
        assert row["strict_correct"] == (
            final == expected if final is not None else False
        )
        assert row["final_available"] == (final is not None)
        assert row["generated_token_count"] == len(row["generated_token_ids"])
        assert 1 <= row["generated_token_count"] <= config["max_new_tokens"]
        assert row["stop_reason"] in ("eos", "token_cap")
        assert row["completed"] == (row["stop_reason"] == "eos")
        assert row["truncated"] == (row["stop_reason"] == "token_cap")
        if row["truncated"]:
            assert row["generated_token_count"] == config["max_new_tokens"]
        assert row["query_tokens"] > 0
        for value in row["timing"].values():
            assert isinstance(value, (int, float)) and value >= 0
        lengths = row["prompt_lengths"]
        c = row["condition"]
        expected_length = (
            lengths["joint"]
            if c == "full_text"
            else (
                lengths["shared_prefix"]
                if c == "none"
                else (
                    lengths["worker_a"]
                    if c in ("only_a", "single_hop_a")
                    else (
                        lengths["worker_b"]
                        if c in ("only_b", "single_hop_b")
                        else lengths["worker_a"]
                        + lengths["worker_b"]
                        - lengths["shared_prefix"]
                    )
                )
            )
        )
        assert row["past_cache_tokens"] == expected_length
    expected = len(config["seeds"]) * config["queries"] * len(config["conditions"])
    if data["status"] == "complete":
        assert len(rows) == expected
        for ci, seed in enumerate(config["seeds"]):
            for qi in range(config["queries"]):
                assert {
                    f"c{ci:04d}-s{seed}/q{qi:03d}/{c}" for c in config["conditions"]
                } <= seen
    for condition, summary in data["summary"]["conditions"].items():
        xs = [r for r in rows if r["condition"] == condition]
        assert summary["n"] == len(xs)
        assert summary["normalized_correct"] == sum(r["normalized_correct"] for r in xs)
        assert summary["truncated"] == sum(r["truncated"] for r in xs)
    report = dict(
        status="passed",
        schema_version=data["schema_version"],
        run_status=data["status"],
        records=len(rows),
        expected_records=expected,
        corpora=len(corpora),
        duplicate_rows=False,
        scoring_recomputed=True,
        cache_lengths_verified=True,
        manifest_fingerprint_verified=True,
    )
    (run / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    print(json.dumps(audit(args.run, args.allow_incomplete), indent=2))


if __name__ == "__main__":
    main()
