import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from experiment import (
    Block,
    Engine,
    append_jsonl,
    body,
    dataset,
    fresh_cache,
    merge,
    read_jsonl,
    rebase_keys,
    run_checks,
    score,
    select_queries,
    SYSTEM,
    NAMES,
    LOCKERS,
    CODES,
)


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        class Result:
            input_ids = torch.tensor([[5 + ord(s) % 50 for s in text]])

        return Result()

    def decode(self, tokens, **kwargs):
        return " ".join(str(t) for t in tokens)


def tiny_config():
    return Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=512,
        eos_token_id=1,
        rope_theta=1000000.0,
        attention_dropout=0.0,
    )


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.rotary = Qwen3RotaryEmbedding(tiny_config())
        self.k = torch.randn(1, 2, 7, 8)
        self.v = torch.randn_like(self.k)

    def test_zero_and_inverse_rebase(self):
        self.assertTrue(torch.equal(rebase_keys(self.k, 0, self.rotary), self.k))
        back = rebase_keys(rebase_keys(self.k, 71, self.rotary), -71, self.rotary)
        torch.testing.assert_close(back, self.k, rtol=0, atol=1e-6)

    def test_matches_actual_qwen_rotary_at_shifted_positions(self):
        pos = torch.arange(7)[None]
        cos, sin = self.rotary(self.k, pos)
        _, original = apply_rotary_pos_emb(self.k, self.k, cos, sin)
        cos, sin = self.rotary(self.k, pos + 19)
        _, reference = apply_rotary_pos_emb(self.k, self.k, cos, sin)
        torch.testing.assert_close(
            rebase_keys(original, 19, self.rotary), reference, rtol=0, atol=2e-6
        )

    def test_bfloat16_rebase_uses_float32_trig(self):
        result = rebase_keys(self.k.bfloat16(), 1001, self.rotary)
        expected = rebase_keys(self.k.bfloat16().float(), 1001, self.rotary).bfloat16()
        self.assertTrue(torch.equal(result, expected))

    def test_prefix_removed_once_and_values_preserved(self):
        a = Block(((self.k.clone(), self.v.clone()),))
        b = body(a, 3)
        merged = merge((a, b), self.rotary)
        self.assertEqual(merged.length, 11)
        torch.testing.assert_close(
            merged.layers[0][0][:, :, 7:], rebase_keys(self.k[:, :, 3:], 4, self.rotary)
        )
        self.assertTrue(
            torch.equal(merged.layers[0][1], torch.cat((self.v, self.v[:, :, 3:]), -2))
        )
        self.assertTrue(torch.equal(a.layers[0][0], self.k))

    def test_fresh_dynamic_cache_does_not_alias_pristine(self):
        a = Block(((self.k, self.v),))
        cache = fresh_cache(a)
        cache.layers[0].keys.add_(1)
        self.assertTrue(torch.equal(a.layers[0][0], self.k))
        self.assertFalse(torch.equal(cache.layers[0].keys, self.k))

    def test_real_tiny_model_generation_and_reload_are_immutable(self):
        model = Qwen3ForCausalLM(tiny_config()).eval()
        model.config._attn_implementation = "eager"
        engine = Engine(model, TinyTokenizer(), "cpu")
        a = engine.prefill("small records")
        copies = tuple((k.clone(), v.clone()) for k, v in a.layers)
        engine.eos = {-1}  # No vocabulary token can terminate this branch.
        result = engine.generate(a, "later query", 4)
        self.assertEqual(result["generated_token_count"], 4)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["completed"])
        engine.eos = set(range(64))  # Every first token terminates this branch.
        terminated = engine.generate(a, "later query", 4)
        self.assertEqual(terminated["generated_token_count"], 1)
        self.assertTrue(terminated["completed"])
        self.assertFalse(terminated["truncated"])
        for (k, v), (x, y) in zip(a.layers, copies):
            self.assertTrue(torch.equal(k, x) and torch.equal(v, y))
        with tempfile.TemporaryDirectory() as tmp:
            checks = run_checks(engine, a, "later query", Path(tmp))
            self.assertEqual(checks["save_reload_logits_max_abs"], 0)


class ProtocolTests(unittest.TestCase):
    def test_dataset_target_independence_and_derangement(self):
        data = dataset(1234, 8)
        self.assertNotIn("query_entry_indices", data)
        self.assertEqual(select_queries(data, 4), select_queries(data, 4))
        for entry in data["entries"]:
            self.assertNotEqual(data["wrong_mapping"][entry["locker"]], entry["code"])
        self.assertEqual(
            [s.split()[1] for s in data["source_b"].splitlines()],
            [s.split()[1] for s in data["wrong_b"].splitlines()],
        )
        self.assertEqual(
            sorted(e["code"] for e in data["entries"]),
            sorted(e["code"] for e in dataset(4321, 8)["entries"]),
        )

    def test_worked_example_entities_do_not_overlap_evidence_pools(self):
        import re

        example = re.search(
            r"if (\w+) uses locker (\w+) and locker \w+ has code (\w+)", SYSTEM
        )
        self.assertIsNotNone(example)
        name, locker, code = example.groups()
        self.assertNotIn(name, NAMES)
        self.assertNotIn(locker, LOCKERS)
        self.assertNotIn(code, CODES)

    def test_never_grade_hidden_reasoning(self):
        self.assertFalse(
            score("reason</think><think>blue", "blue", True, "code")[
                "normalized_correct"
            ]
        )
        self.assertFalse(
            score("I think blue is correct", "blue", True, "code")["normalized_correct"]
        )
        self.assertIsNone(score("<think>blue", "blue", False, "code")["final_text"])
        self.assertTrue(
            score("blue?</think>red", "red", True, "code")["normalized_correct"]
        )
        self.assertTrue(
            score("The code is RED.", "red", False, "code")["normalized_correct"]
        )
        self.assertFalse(
            score("red or blue", "red", False, "code")["normalized_correct"]
        )
        self.assertTrue(score("Locker G.", "G", False, "locker")["normalized_correct"])

    def test_jsonl_torn_tail_repair_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            append_jsonl(path, {"record_id": "one"})
            with path.open("ab") as f:
                f.write(b'{"record_id":')
            with self.assertRaises(ValueError):
                read_jsonl(path)
            self.assertEqual(read_jsonl(path, repair_tail=True), [{"record_id": "one"}])
            self.assertEqual(len(list(Path(tmp).glob("*.bin"))), 1)
            append_jsonl(path, {"record_id": "two"})
            self.assertEqual(len(read_jsonl(path)), 2)

    def test_jsonl_middle_corruption_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text("bad\n" + json.dumps({"ok": 1}) + "\n")
            with self.assertRaises(ValueError):
                read_jsonl(path, repair_tail=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
