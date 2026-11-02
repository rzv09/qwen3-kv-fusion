#!/usr/bin/env python3
"""Same-checkpoint, query-independent KV composition. No scheduler or remote code."""
import argparse
import hashlib
import inspect
import importlib.metadata
import json
import os
import platform
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

SCHEMA = "1.0.0"
PROTOCOL = "qwen3-independent-kv-v1"
CONDITIONS = (
    "full_text",
    "none",
    "only_a",
    "only_b",
    "merged_ab",
    "merged_ba",
    "wrong_partner",
    "single_hop_a",
    "single_hop_b",
)
CODES = (
    "red blue green gold pink white black orange purple silver brown cyan lime "
    "navy teal beige coral ivory maroon olive peach violet indigo amber"
).split()
NAMES = (
    "Ada Bela Cora Dina Evan Faye Gail Hugo Iris Jude Kira Lena Mina Nora Omar "
    "Pia Quin Rosa Seth Tara Uma Vera Will Xena Yuri Zola Alma Bess Cleo Dana "
    "Elsa Finn Gwen Hana Ivan Jade Karl Leah Milo Nia Orla Paul Rina Sara Theo "
    "Vera Walt Yara Zane"
).split()
# No I or O: avoid ambiguity in letter-only answers.
LOCKERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
SYSTEM = (
    "Use the records to answer the question. Follow person to locker, then locker "
    "to code. For example: if ExamplePerson uses locker EXAMPLE and locker EXAMPLE has code examplecode, "
    "ExamplePerson has code examplecode. Carefully look up the person, then look up that exact "
    "locker. Think briefly, using fewer than 50 words. Reply with only the answer. "
    "If the records do not establish the answer, reply UNKNOWN."
)
PREFIX = "<|im_start|>system\n" + SYSTEM + "\n<|im_end|>\n<|im_start|>user\nRecords:\n"
ROOT = Path(__file__).resolve().parent


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path, repair_tail=False):
    """Only a torn final line can be removed; preserve it separately for inspection."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("missing final newline")
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError) as e:
            if not repair_tail or i != len(lines) - 1:
                raise ValueError(
                    f"Invalid JSONL at {path}:{i+1}; refusing to skip data"
                ) from e
            backup = path.with_name(path.name + f".torn-tail-{time.time_ns()}.bin")
            backup.write_bytes(data[offset:])
            with path.open("r+b") as f:
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())
            print(
                f"Restored complete JSONL prefix; preserved torn tail: {backup}",
                flush=True,
            )
            break
        offset += len(line)
    return rows


def dataset(seed, chains):
    r = random.Random(seed)
    names = r.sample(sorted(set(NAMES)), chains)
    lockers = r.sample(LOCKERS, chains)
    codes = CODES[:chains].copy()
    r.shuffle(codes)
    entries = [
        dict(person=name, locker=locker, code=code)
        for name, locker, code in zip(names, lockers, codes)
    ]
    order_a, order_b = list(range(chains)), list(range(chains))
    r.shuffle(order_a)
    r.shuffle(order_b)
    wrong = {lockers[i]: codes[(i + 1) % chains] for i in range(chains)}
    a = "\n".join(f"{names[i]} uses locker {lockers[i]}." for i in order_a) + "\n"
    b = "\n".join(f"Locker {lockers[i]} has code {codes[i]}." for i in order_b) + "\n"
    wrong_b = (
        "\n".join(f"Locker {lockers[i]} has code {wrong[lockers[i]]}." for i in order_b)
        + "\n"
    )
    # No target selection here. Target RNG is instantiated only after cache creation.
    return dict(
        seed=seed,
        entries=entries,
        order_a=order_a,
        order_b=order_b,
        source_a=a,
        source_b=b,
        wrong_b=wrong_b,
        wrong_mapping=wrong,
        target_selection_seed=seed ^ 0x5EEDC0DE,
    )


def select_queries(corpus, count):
    r = random.Random(corpus["target_selection_seed"])
    return r.sample(range(len(corpus["entries"])), count)


@dataclass(frozen=True)
class Block:
    layers: tuple
    source_start: int = 0

    @property
    def length(self):
        return self.layers[0][0].shape[-2]


def body(block, prefix_len):
    return Block(
        tuple(
            (k[:, :, prefix_len:, :], v[:, :, prefix_len:, :]) for k, v in block.layers
        ),
        prefix_len,
    )


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rebase_keys(keys, shift, rotary):
    """Actual Qwen3 rotary module, incremental shift of already rotated keys."""
    if shift == 0:
        return keys.clone()
    # Qwen's rotary returns the dummy input's dtype. Float32 trig and arithmetic
    # avoid low-precision phase error; cast back once to the original cache dtype.
    dummy = torch.empty((1, 1, keys.shape[-1]), dtype=torch.float32, device=keys.device)
    positions = torch.tensor([[shift]], device=keys.device, dtype=torch.long)
    cos, sin = rotary(dummy, positions)
    x = keys.float()
    return (x * cos[:, None, :, :] + rotate_half(x) * sin[:, None, :, :]).to(keys.dtype)


def merge(blocks, rotary, rebase=True):
    layers = []
    offset = 0
    for block in blocks:
        shifted = tuple(
            (
                (
                    rebase_keys(k, offset - block.source_start, rotary)
                    if rebase
                    else k.clone()
                ),
                v.clone(),
            )
            for k, v in block.layers
        )
        if not layers:
            layers = list(shifted)
        else:
            layers = [
                (torch.cat((a, k), dim=-2), torch.cat((b, v), dim=-2))
                for (a, b), (k, v) in zip(layers, shifted)
            ]
        offset += block.length
    return Block(tuple(layers))


def fresh_cache(block):
    cache = DynamicCache()
    for i, (k, v) in enumerate(block.layers):
        cache.update(k.clone(), v.clone(), i)
    return cache


def tensor_versions(block):
    return [(k._version, v._version, k.shape, v.shape) for k, v in block.layers]


def final_answer(text, thinking):
    # With thinking enabled the opening tag is already in the prompt, so a
    # missing closing tag means ALL generated output is hidden reasoning.
    if "</think>" in text:
        suffix = text.rsplit("</think>", 1)[1].strip()
        if "<think>" in suffix:
            return None, False
        return suffix, True
    if thinking or "<think>" in text:
        return None, False
    return text.strip(), True


def score(text, expected, thinking, kind):
    final, available = final_answer(text, thinking)
    parsed = None
    if available and final:
        if kind == "locker":
            clean = final.strip(" \t\n.*`")
            found = re.findall(r"\blocker\s+([A-Z])\b", final, flags=re.IGNORECASE)
            found = {s.upper() for s in found if s.upper() in LOCKERS}
            if clean.upper() in LOCKERS and len(clean) == 1:
                parsed = clean.upper()
            elif len(found) == 1:
                parsed = next(iter(found))
        else:
            found = set(re.findall(r"\b(?:" + "|".join(CODES) + r")\b", final.lower()))
            if len(found) == 1:
                parsed = next(iter(found))
    return dict(
        final_text=final,
        final_available=available,
        parsed_answer=parsed,
        strict_correct=final == expected if available else False,
        normalized_correct=parsed is not None and parsed.lower() == expected.lower(),
    )


def query_text(entry, condition, thinking):
    if condition == "single_hop_a":
        question = (
            f"Which locker does {entry['person']} use? Reply with only its letter."
        )
        expected, kind = entry["locker"], "locker"
    elif condition == "single_hop_b":
        question = (
            f"What is the code for locker {entry['locker']}? Reply with only the code."
        )
        expected, kind = entry["code"], "code"
    else:
        question = f"What is the code for {entry['person']}? Reply with only the code."
        expected, kind = entry["code"], "code"
    suffix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return (
        "\nQuestion: " + question + "\n<|im_end|>\n<|im_start|>assistant\n" + suffix,
        expected,
        kind,
    )


def normalize_device(device):
    """Resolve bare CUDA to the current logical index; preserve explicit indices."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


class Engine:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = (
            model,
            tokenizer,
            normalize_device(device),
        )
        self.rotary = model.model.rotary_emb
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.add(tokenizer.eos_token_id)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timed(self, fn):
        self.sync()
        start = time.perf_counter()
        result = fn()
        self.sync()
        return result, time.perf_counter() - start

    def ids(self, text):
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

    @torch.no_grad()
    def prefill(self, text):
        ids = self.ids(text)
        pos = torch.arange(ids.shape[1], device=self.device)
        result = self.model(
            ids,
            attention_mask=torch.ones_like(ids),
            position_ids=pos[None],
            cache_position=pos,
            use_cache=True,
        )
        return Block(
            tuple(
                (layer.keys.detach(), layer.values.detach())
                for layer in result.past_key_values.layers
            )
        )

    @torch.no_grad()
    def first_logits(self, block, text):
        cache = fresh_cache(block)
        ids = self.ids(text)
        n = block.length
        positions = torch.arange(n, n + ids.shape[1], device=self.device)
        result = self.model(
            ids,
            past_key_values=cache,
            attention_mask=torch.ones(
                (1, n + ids.shape[1]), dtype=torch.long, device=self.device
            ),
            position_ids=positions[None],
            cache_position=positions,
            use_cache=True,
        )
        return result.logits[:, -1, :].detach().cpu()

    @torch.no_grad()
    def generate(self, block, text, max_tokens):
        pristine = tensor_versions(block)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        all_start = time.perf_counter()
        cache, clone_seconds = self.timed(lambda: fresh_cache(block))
        ids, tokenize_seconds = self.timed(lambda: self.ids(text))
        query_len = ids.shape[1]
        n, tokens = block.length, []
        self.sync()
        start = time.perf_counter()
        question_seconds = 0.0
        stop = "token_cap"
        for step in range(max_tokens):
            positions = torch.arange(n, n + ids.shape[1], device=self.device)
            result = self.model(
                ids,
                past_key_values=cache,
                attention_mask=torch.ones(
                    (1, n + ids.shape[1]), dtype=torch.long, device=self.device
                ),
                position_ids=positions[None],
                cache_position=positions,
                use_cache=True,
            )
            next_id = int(result.logits[0, -1].argmax().item())
            tokens.append(next_id)
            n += ids.shape[1]
            if step == 0:
                self.sync()
                question_seconds = time.perf_counter() - start
            if next_id in self.eos:
                stop = "eos"
                break
            ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        self.sync()
        generation_seconds = time.perf_counter() - start
        raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
        text_out = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.sync()
        total = time.perf_counter() - all_start
        assert (
            tensor_versions(block) == pristine
        ), "Input cache mutated during generation"
        memory = (
            dict(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            )
            if self.device.type == "cuda"
            else None
        )
        return dict(
            generated_text=text_out,
            generated_text_with_special_tokens=raw,
            generated_token_ids=tokens,
            generated_token_count=len(tokens),
            stop_reason=stop,
            completed=stop == "eos",
            truncated=stop == "token_cap",
            query_tokens=query_len,
            past_cache_tokens=block.length,
            timing=dict(
                cache_clone_seconds=clone_seconds,
                query_tokenize_seconds=tokenize_seconds,
                question_forward_and_first_token_seconds=question_seconds,
                generation_including_question_seconds=generation_seconds,
                decode_after_first_token_seconds=generation_seconds - question_seconds,
                receiver_total_seconds=total,
            ),
            memory=memory,
        )


def resolve_identity(args):
    if args.local_model:
        path = Path(args.local_model).resolve()
        if not path.is_dir():
            raise ValueError(f"Local model directory missing: {path}")
        selected = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".safetensors", ".txt", ".jinja", ".model")
        )
        if not any(p.suffix == ".safetensors" for p in selected):
            raise ValueError(
                "Local model must contain safetensors weights; remote custom code is disabled"
            )
        files = {str(p.relative_to(path)): file_hash(p) for p in selected}
        revision = "local-sha256:" + stable_hash(files)
        return str(path), dict(
            model_id=args.model,
            tokenizer_id=args.model,
            model_resolved_revision=revision,
            tokenizer_resolved_revision=revision,
            declared_revision=args.revision,
            local_content_hashes=files,
            source="local",
            source_path=str(path),
        )
    sha = HfApi().model_info(args.model, revision=args.revision or "main").sha
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hub did not return an immutable commit SHA")
    return args.model, dict(
        model_id=args.model,
        tokenizer_id=args.model,
        model_resolved_revision=sha,
        tokenizer_resolved_revision=sha,
        declared_revision=args.revision,
        source="huggingface",
    )


def runtime(device):
    device = normalize_device(device)
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        architecture=platform.machine(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda_runtime=torch.version.cuda,
        device=str(device),
        packages={
            n: importlib.metadata.version(n)
            for n in ("huggingface-hub", "tokenizers", "safetensors", "numpy")
        },
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = dict(
            name=props.name,
            total_memory_bytes=props.total_memory,
            capability=list(torch.cuda.get_device_capability(device)),
        )
    return result


def run_checks(engine, a, query, out):
    versions = tensor_versions(a)
    reference = engine.first_logits(a, query)
    fd, temporary = tempfile.mkstemp(prefix="runtime-check-", suffix=".pt", dir=out)
    os.close(fd)
    try:
        torch.save(a.layers, temporary)
        loaded = Block(
            tuple(torch.load(temporary, map_location=engine.device, weights_only=True))
        )
        for (k, v), (x, y) in zip(a.layers, loaded.layers):
            assert torch.equal(k, x) and torch.equal(v, y)
        actual = engine.first_logits(loaded, query)
        error = float((reference - actual).abs().max())
        assert torch.equal(reference, actual), f"Save/reload logits mismatch: {error}"
    finally:
        Path(temporary).unlink(missing_ok=True)
    assert tensor_versions(a) == versions
    # Small reference fixture is explicitly optional; whole caches are large.
    return dict(
        save_reload_tensors_exact=True,
        save_reload_logits_max_abs=error,
        original_tensor_versions_unchanged=True,
    )


def build_condition(name, base, engine, prefix_len):
    if name == "full_text":
        return base["joint"]
    if name == "none":
        return base["prefix"]
    if name in ("only_a", "single_hop_a"):
        return base["a"]
    if name in ("only_b", "single_hop_b"):
        return base["b"]
    a, b = body(base["a"], prefix_len), body(base["b"], prefix_len)
    if name == "merged_ab":
        return merge((base["a"], b), engine.rotary)
    if name == "merged_ba":
        return merge((base["prefix"], b, a), engine.rotary)
    if name == "wrong_partner":
        return merge((base["a"], body(base["wrong"], prefix_len)), engine.rotary)
    if name == "merged_ab_no_rebase":
        return merge((base["a"], b), engine.rotary, rebase=False)
    raise ValueError(name)


def consolidate(out, manifest, corpora, rows, complete):
    summary = {}
    for condition in manifest["config"]["conditions"]:
        xs = [x for x in rows if x["condition"] == condition]
        summary[condition] = dict(
            n=len(xs),
            normalized_correct=sum(x["normalized_correct"] for x in xs),
            strict_correct=sum(x["strict_correct"] for x in xs),
            completed_eos=sum(x["completed"] for x in xs),
            truncated=sum(x["truncated"] for x in xs),
            final_unavailable=sum(not x["final_available"] for x in xs),
            eos_and_correct=sum(x["completed"] and x["normalized_correct"] for x in xs),
            mean_generation_seconds=(
                sum(x["timing"]["generation_including_question_seconds"] for x in xs)
                / len(xs)
                if xs
                else None
            ),
        )
    status = "complete" if complete else "incomplete"
    expected = (
        len(manifest["config"]["seeds"])
        * manifest["config"]["queries"]
        * len(manifest["config"]["conditions"])
    )
    report = dict(
        schema_version=SCHEMA,
        run_fingerprint=manifest["fingerprint"],
        status=status,
        expected_records=expected,
        observed_records=len(rows),
        conditions=summary,
    )
    atomic_json(out / "summary.json", report)
    atomic_json(
        out / "results.json",
        dict(
            schema_version=SCHEMA,
            status=status,
            manifest=manifest,
            corpora=corpora,
            results=rows,
            summary=report,
        ),
    )
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument(
        "--local-model",
        help="Existing local safetensors model/tokenizer directory; no network",
    )
    p.add_argument(
        "--revision", help="Hub revision/tag/commit, resolved and pinned at startup"
    )
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    p.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument(
        "--seeds",
        help="Explicit comma-separated corpus seeds; overrides --seed and --corpora",
    )
    p.add_argument("--corpora", type=int, default=50)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--thinking", choices=["on", "off"], default="on")
    p.add_argument(
        "--single-hop-thinking", choices=["same", "on", "off"], default="same"
    )
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    p.add_argument(
        "--threads", type=int, default=5, help="CPU threads; recorded even on CUDA"
    )
    p.add_argument("--no-rebase-ablation", action="store_true")
    p.add_argument(
        "--export-handoff",
        action="store_true",
        help="Save first A cache + later query/logits for optional fresh-process check",
    )
    p.add_argument(
        "--save-caches",
        action="store_true",
        help="Large per-corpus tensor files; off by default",
    )
    p.add_argument(
        "--resume", action="store_true", help="Require exact matching manifest/protocol"
    )
    p.add_argument("--out", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if not 2 <= args.chains <= min(len(CODES), len(LOCKERS)):
        raise ValueError("chains must be between 2 and 24")
    if (
        not 1 <= args.queries <= args.chains
        or args.max_new_tokens < 1
        or args.corpora < 1
    ):
        raise ValueError(
            "Require 1 <= queries <= chains, positive corpora/token budget"
        )
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = random.Random(args.seed).sample(range(1, 2**31), args.corpora)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Corpus seeds must be nonempty and unique")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu/cuda tested by this script")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Use the cluster CUDA-enabled PyTorch environment; requirements do not install torch."
        )
    if device.type == "cuda":
        device = normalize_device(device)
        torch.cuda.set_device(device)
    if (
        device.type == "cuda"
        and args.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("CUDA device does not report BF16 support")
    out = args.out.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(
            "Output exists; choose a new path or explicitly --resume matching run"
        )
    if args.resume and not (out / "manifest.json").is_file():
        raise FileNotFoundError("--resume requires an existing manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    # Advisory lock works on Linux and macOS; prevents two writers to one run.
    import fcntl

    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another process holds this run directory") from e
        run(args, seeds, device, out)


def run(args, seeds, device, out):
    # Resume remote main at the ORIGINAL commit, even if the remote branch moved.
    previous = json.loads((out / "manifest.json").read_text()) if args.resume else None
    if previous and not args.local_model and not args.revision:
        args.revision = previous["model"]["model_resolved_revision"]
    model_source, identity = resolve_identity(args)
    config = dict(
        model=args.model,
        local_model=str(Path(args.local_model).resolve()) if args.local_model else None,
        resolved_revision=identity["model_resolved_revision"],
        device=str(device),
        dtype=args.dtype,
        seeds=seeds,
        chains=args.chains,
        queries=args.queries,
        max_new_tokens=args.max_new_tokens,
        thinking=args.thinking,
        single_hop_thinking=args.single_hop_thinking,
        attention=args.attention,
        threads=args.threads,
        save_caches=args.save_caches,
        export_handoff=args.export_handoff,
        conditions=list(CONDITIONS)
        + (["merged_ab_no_rebase"] if args.no_rebase_ablation else []),
    )
    environment = runtime(device)
    source_hash = file_hash(Path(__file__))
    fingerprint = stable_hash(
        dict(
            schema=SCHEMA,
            protocol=PROTOCOL,
            config=config,
            source_sha256=source_hash,
            torch=environment["torch"],
            transformers=environment["transformers"],
            cuda=environment["cuda_runtime"],
            packages=environment["packages"],
            gpu=environment.get("gpu"),
        )
    )
    manifest = dict(
        schema_version=SCHEMA,
        protocol=PROTOCOL,
        fingerprint=fingerprint,
        config=config,
        model=identity,
        runtime=environment,
        source_sha256=source_hash,
        shared_prefix_text=PREFIX,
        timing_scope="CUDA synchronized wall time; generation includes question pass; clone/tokenize/merge/prefill separate",
        worker_execution="One immutable loaded model reused sequentially; not a network throughput benchmark",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if previous:
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                "Resume fingerprint mismatch: config, model content/revision, code, or runtime versions changed. Use a new output directory."
            )
        manifest = previous
    else:
        atomic_json(out / "manifest.json", manifest)
    rows = read_jsonl(out / "records.jsonl", repair_tail=args.resume)
    corpora = read_jsonl(out / "corpora.jsonl", repair_tail=args.resume)
    existing_ids = [r["record_id"] for r in rows]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Duplicate record IDs already present; refusing resume")
    if any(r["run_fingerprint"] != fingerprint for r in rows):
        raise ValueError("Records do not match run fingerprint")
    done = set(existing_ids)
    corpus_by_id = {c["corpus_id"]: c for c in corpora}
    if len(corpus_by_id) != len(corpora):
        raise ValueError("Duplicate corpus metadata")
    expected_count = len(seeds) * args.queries * len(config["conditions"])
    valid_ids = {
        f"c{ci:04d}-s{seed}/q{qi:03d}/{condition}"
        for ci, seed in enumerate(seeds)
        for qi in range(args.queries)
        for condition in config["conditions"]
    }
    if not done <= valid_ids:
        raise ValueError("Unrecognized record IDs in existing run")
    consolidate(out, manifest, corpora, rows, False)
    if len(rows) == expected_count:
        consolidate(out, manifest, corpora, rows, True)
        print("Run already complete; no model loaded and no records repeated.")
        return
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    # Stable single-device math; not a promise of bitwise cross-platform identity.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    load_kwargs = dict(local_files_only=bool(args.local_model), trust_remote_code=False)
    if not args.local_model:
        load_kwargs["revision"] = identity["model_resolved_revision"]
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    if tokenizer.convert_tokens_to_ids("<think>") == tokenizer.unk_token_id:
        raise ValueError(
            "This experiment requires a Qwen3 tokenizer with thinking tags"
        )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention,
            **load_kwargs,
        )
        .to(device)
        .eval()
    )
    if model.config.model_type != "qwen3" or model.config.rope_scaling is not None:
        raise ValueError(
            "Only standard unscaled Qwen3 RoPE supported; refuse silent extrapolation"
        )
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("Sliding-window cache layout is unsupported")
    engine = Engine(model, tokenizer, device)
    engine.sync()
    model_load_seconds = time.perf_counter() - start
    manifest.setdefault(
        "loaded_model",
        dict(
            model_class=type(model).__name__,
            tokenizer_class=type(tokenizer).__name__,
            model_config=model.config.to_dict(),
        ),
    )
    manifest.setdefault("initial_model_load_seconds", model_load_seconds)
    atomic_json(out / "manifest.json", manifest)
    atomic_json(
        out / "load_metadata.json",
        dict(
            model_load_seconds=model_load_seconds,
            tokenizer_class=type(tokenizer).__name__,
            model_class=type(model).__name__,
            model_config=model.config.to_dict(),
        ),
    )
    (out / "rotary_source.py.txt").write_text(
        inspect.getsource(type(model.model.rotary_emb).forward)
    )
    (out / "attention_source.py.txt").write_text(
        inspect.getsource(type(model.model.layers[0].self_attn).forward)
    )
    try:
        for corpus_index, seed in enumerate(seeds):
            corpus_id = f"c{corpus_index:04d}-s{seed}"
            wanted = {
                f"{corpus_id}/q{q:03d}/{c}"
                for q in range(args.queries)
                for c in config["conditions"]
            }
            if wanted <= done:
                continue
            data = dataset(seed, args.chains)
            base, phases = {}, {}
            for name, text in [
                ("a", PREFIX + data["source_a"]),
                ("b", PREFIX + data["source_b"]),
                ("wrong", PREFIX + data["wrong_b"]),
                ("joint", PREFIX + data["source_a"] + data["source_b"]),
                ("prefix", PREFIX),
            ]:
                base[name], phases[name + "_prefill_seconds"] = engine.timed(
                    lambda text=text: engine.prefill(text)
                )
            versions = {k: tensor_versions(b) for k, b in base.items()}
            prefix_len = base["prefix"].length
            # First target sampling happens HERE, after ALL independent caches exist.
            chosen = select_queries(data, args.queries)
            query_entries = [data["entries"][i] for i in chosen]
            stored = dict(
                schema_version=SCHEMA,
                run_fingerprint=fingerprint,
                corpus_id=corpus_id,
                corpus_index=corpus_index,
                **data,
                query_entry_indices=chosen,
                source_sha256=stable_hash(data),
                cache_lengths={k: b.length for k, b in base.items()},
                prefill_timing=phases,
                targets_sampled_after_cache_creation=True,
            )
            if corpus_id in corpus_by_id:
                old = corpus_by_id[corpus_id]
                for key in (
                    "source_sha256",
                    "query_entry_indices",
                    "cache_lengths",
                    "run_fingerprint",
                ):
                    if old[key] != stored[key]:
                        raise ValueError(f"Rebuilt corpus differs on resume: {key}")
                stored = old  # Original prefill timing remains authoritative for earlier rows.
            else:
                append_jsonl(out / "corpora.jsonl", stored)
                corpora.append(stored)
                corpus_by_id[corpus_id] = stored
            if args.save_caches:
                for name in ("a", "b"):
                    path = out / f"{corpus_id}-{name}.pt"
                    if not path.exists():
                        torch.save(base[name].layers, path)
            if corpus_index == 0:
                q, _, _ = query_text(query_entries[0], "only_a", args.thinking == "on")
                check, seconds = engine.timed(
                    lambda: run_checks(engine, base["a"], q, out)
                )
                check["seconds"] = seconds
                atomic_json(out / "runtime_checks.json", check)
                if args.export_handoff:
                    torch.save(base["a"].layers, out / "handoff-a.pt")
                    torch.save(
                        engine.first_logits(base["a"], q), out / "handoff-logits.pt"
                    )
                    atomic_json(
                        out / "handoff.json",
                        dict(
                            cache="handoff-a.pt",
                            reference="handoff-logits.pt",
                            query=q,
                            corpus_id=corpus_id,
                        ),
                    )
            for qi, entry in enumerate(query_entries):
                for condition in config["conditions"]:
                    record_id = f"{corpus_id}/q{qi:03d}/{condition}"
                    if record_id in done:
                        continue
                    thinking = args.thinking == "on"
                    if (
                        condition.startswith("single_hop")
                        and args.single_hop_thinking != "same"
                    ):
                        thinking = args.single_hop_thinking == "on"
                    q, expected, kind = query_text(entry, condition, thinking)
                    block, merge_seconds = engine.timed(
                        lambda: build_condition(condition, base, engine, prefix_len)
                    )
                    if (
                        block.length + len(engine.ids(q)[0]) + args.max_new_tokens
                        > model.config.max_position_embeddings
                    ):
                        raise ValueError(
                            "Requested context plus token budget exceeds model max_position_embeddings"
                        )
                    generated = engine.generate(block, q, args.max_new_tokens)
                    generated["timing"]["condition_cache_build_seconds"] = merge_seconds
                    parsed = score(
                        generated["generated_text"], expected, thinking, kind
                    )
                    row = dict(
                        schema_version=SCHEMA,
                        run_fingerprint=fingerprint,
                        record_id=record_id,
                        corpus_id=corpus_id,
                        corpus_index=corpus_index,
                        corpus_seed=seed,
                        query_id=f"q{qi:03d}",
                        query_entry_index=chosen[qi],
                        person=entry["person"],
                        condition=condition,
                        question_text=q,
                        thinking=thinking,
                        expected_answer=expected,
                        answer_kind=kind,
                        prompt_lengths=dict(
                            shared_prefix=prefix_len,
                            worker_a=base["a"].length,
                            worker_b=base["b"].length,
                            joint=base["joint"].length,
                        ),
                        prefill_timing_this_attempt=phases,
                        **generated,
                        **parsed,
                    )
                    append_jsonl(out / "records.jsonl", row)
                    rows.append(row)
                    done.add(record_id)
                    # Atomic portable checkpoints after every row; partial runs stay explicit.
                    consolidate(
                        out, manifest, corpora, rows, len(done) == expected_count
                    )
                    print(
                        f"{len(done)}/{expected_count} {record_id} expected={expected} parsed={parsed['parsed_answer']} stop={generated['stop_reason']}",
                        flush=True,
                    )
                    del block
            assert all(
                tensor_versions(base[k]) == versions[k] for k in base
            ), "Pristine worker cache modified"
            base.clear()
    except BaseException as e:
        consolidate(out, manifest, corpora, rows, False)
        atomic_json(
            out / "last_interruption.json",
            dict(
                type=type(e).__name__,
                message=str(e),
                records=len(rows),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
        raise
    report = consolidate(out, manifest, corpora, rows, len(done) == expected_count)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
