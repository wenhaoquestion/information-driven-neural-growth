"""Small invariant checks for the phase-six runner; no confirmation experiments.

Run from this directory with ``python -m unittest -v test_frontier``.
All generated datasets here are synthetic fixtures of at most a few dozen rows.
"""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

import phase5_reference as reference
import frontier


class ReferenceMathematicsTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_for_every_parameter(self):
        rng = np.random.default_rng(803)
        x = rng.normal(size=(11, 4))
        y = rng.normal(size=11)
        p = {
            "w": rng.normal(size=(4, 3)),
            "a": np.asarray([0.45, -0.8, 0.2]),
            "c": np.asarray(0.3),
        }
        analytic = reference.grad(x, y, p)
        eps = 1e-6
        for key, value in p.items():
            numerical = np.empty_like(value)
            for ix in np.ndindex(value.shape):
                plus, minus = copy.deepcopy(p), copy.deepcopy(p)
                plus[key][ix] += eps
                minus[key][ix] -= eps
                numerical[ix] = (
                    np.mean((reference.predict(x, plus) - y) ** 2)
                    - np.mean((reference.predict(x, minus) - y) ** 2)
                ) / (2 * eps)
            with self.subTest(parameter=key):
                np.testing.assert_allclose(analytic[key], numerical,
                                           rtol=2e-6, atol=2e-6)

    def test_birth_preserves_function_and_existing_optimizer_state(self):
        s = reference.init(24, 4, 2)
        reference.adam(s, {k: np.ones_like(v) for k, v in s["p"].items()},
                       lr=0.01, clip=100)
        old = copy.deepcopy(s)
        x = np.random.default_rng(932).normal(size=(19, 4))
        w = np.asarray([0.5, -0.5, 0.5, -0.5])
        grown = reference.birth(s, w)
        np.testing.assert_allclose(reference.predict(x, grown["p"]),
                                   reference.predict(x, old["p"]),
                                   rtol=1e-14, atol=1e-14)
        self.assertEqual(grown["step"], old["step"])
        for group in ("p", "m", "v"):
            for key in old[group]:
                np.testing.assert_array_equal(s[group][key], old[group][key])
                self.assertFalse(np.shares_memory(s[group][key], grown[group][key]))
            np.testing.assert_array_equal(grown[group]["w"][:, :-1], old[group]["w"])
            np.testing.assert_array_equal(grown[group]["a"][:-1], old[group]["a"])
            self.assertEqual(float(grown[group]["c"]), float(old[group]["c"]))
            np.testing.assert_array_equal(grown[group]["w"][:, -1],
                                          w if group == "p" else np.zeros(4))
            self.assertEqual(grown[group]["a"][-1], 0)


class SeedingTests(unittest.TestCase):
    def small_config(self):
        cfg = frontier.default_config()
        cfg.update(d=4, arrivals=3, block=5, neval=8)
        return cfg

    def test_named_seed_is_deterministic_and_domain_separated(self):
        first = frontier.named_seed(27, "lowrank_noiseless", "teacher", 2)
        self.assertEqual(first, frontier.named_seed(27, "lowrank_noiseless", "teacher", 2))
        values = {
            first,
            frontier.named_seed(28, "lowrank_noiseless", "teacher", 2),
            frontier.named_seed(27, "indefinite_noisy", "teacher", 2),
            frontier.named_seed(27, "lowrank_noiseless", "stream", 2),
            frontier.named_seed(27, "lowrank_noiseless", "teacher", 3),
        }
        self.assertEqual(len(values), 5)

    def test_stream_and_teacher_repeat_exactly_for_same_seed(self):
        cfg = self.small_config()
        for task in ("null", "lowrank_noiseless", "indefinite_noisy"):
            with self.subTest(task=task):
                x1, y1, spec1 = frontier.generate(101, task, cfg)
                x2, y2, spec2 = frontier.generate(101, task, cfg)
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_array_equal(y1, y2)
                np.testing.assert_array_equal(spec1["matrix"], spec2["matrix"])
                np.testing.assert_array_equal(spec1["rotation"], spec2["rotation"])
                self.assertEqual(spec1["noise"], spec2["noise"])

    def test_teacher_changes_with_seed_without_changing_its_spectrum(self):
        cfg = self.small_config()
        for task in ("lowrank_noiseless", "indefinite_noisy"):
            with self.subTest(task=task):
                _, _, one = frontier.generate(101, task, cfg)
                _, _, two = frontier.generate(102, task, cfg)
                self.assertFalse(np.allclose(one["rotation"], two["rotation"]))
                self.assertFalse(np.allclose(one["matrix"], two["matrix"]))
                np.testing.assert_allclose(np.linalg.eigvalsh(one["matrix"]),
                                           np.linalg.eigvalsh(two["matrix"]),
                                           rtol=1e-13, atol=1e-13)


class OptimizerAgeTests(unittest.TestCase):
    @staticmethod
    def aged_state(mode, prior_steps=9):
        s = frontier.new_state(36, "lowrank_noiseless", 4, 2, mode)
        s["step"] = prior_steps
        for key in s["p"]:
            s["m"][key][...] = 0.2
            s["v"][key][...] = 0.04
            s["age"][key][...] = prior_steps
        return s

    def test_grow_preserves_state_and_function_and_starts_new_ages_at_zero(self):
        x = np.random.default_rng(9).normal(size=(13, 4))
        w = np.asarray([0.5, -0.5, 0.5, -0.5])
        for mode in ("global", "local"):
            with self.subTest(mode=mode):
                s = self.aged_state(mode)
                frozen = copy.deepcopy(s)
                grown = frontier.grow(s, w)
                self.assertEqual(grown["step"], 9)
                np.testing.assert_allclose(reference.predict(x, grown["p"]),
                                           reference.predict(x, s["p"]),
                                           rtol=1e-14, atol=1e-14)
                for group in ("p", "m", "v", "age"):
                    for key in s[group]:
                        np.testing.assert_array_equal(s[group][key], frozen[group][key])
                        self.assertFalse(np.shares_memory(s[group][key], grown[group][key]))
                    np.testing.assert_array_equal(grown[group]["w"][:, :-1], s[group]["w"])
                    np.testing.assert_array_equal(grown[group]["a"][:-1], s[group]["a"])
                    np.testing.assert_array_equal(grown[group]["c"], s[group]["c"])
                    np.testing.assert_array_equal(grown[group]["w"][:, -1],
                                                  w if group == "p" else np.zeros(4))
                    self.assertEqual(grown[group]["a"][-1], 0)

    def test_newborn_global_and_local_bias_corrections_have_independent_expected_updates(self):
        lr, eps = 0.03, 1e-8
        for prior_steps in (9, 11):
            for mode in ("global", "local"):
                with self.subTest(mode=mode, prior_steps=prior_steps):
                    s = frontier.grow(self.aged_state(mode, prior_steps),
                                      np.asarray([0.5, -0.5, 0.5, -0.5]))
                    before = copy.deepcopy(s)
                    g = {key: np.zeros_like(value) for key, value in s["p"].items()}
                    g["w"][:, -1] = 1
                    g["a"][-1] = 1
                    self.assertEqual(frontier.adam(s, g, lr, 100, mode), 0)
                    self.assertEqual(s["step"], prior_steps + 1)
                    t = 1 if mode == "local" else prior_steps + 1
                    newborn_delta = lr * (0.1 / (1 - 0.9 ** t)) / (
                        np.sqrt(0.001 / (1 - 0.999 ** t)) + eps)
                    old_delta = lr * (0.18 / (1 - 0.9 ** (prior_steps + 1))) / (
                        np.sqrt(0.03996 / (1 - 0.999 ** (prior_steps + 1))) + eps)
                    for key in ("w", "a"):
                        np.testing.assert_allclose(before["p"][key][..., -1] - s["p"][key][..., -1],
                                                   newborn_delta, rtol=1e-12, atol=1e-14)
                        np.testing.assert_allclose(before["p"][key][..., :-1] - s["p"][key][..., :-1],
                                                   old_delta, rtol=1e-12, atol=1e-14)
                        np.testing.assert_allclose(s["m"][key][..., -1], 0.1)
                        np.testing.assert_allclose(s["v"][key][..., -1], 0.001)
                        np.testing.assert_allclose(s["m"][key][..., :-1], 0.18)
                        np.testing.assert_allclose(s["v"][key][..., :-1], 0.03996)
                        if mode == "local":
                            np.testing.assert_array_equal(s["age"][key][..., -1], 1)
                            np.testing.assert_array_equal(s["age"][key][..., :-1], prior_steps + 1)
                    np.testing.assert_allclose(before["p"]["c"] - s["p"]["c"], old_delta)
                    if mode == "local":
                        self.assertEqual(int(s["age"]["c"]), prior_steps + 1)
                        # A second update must advance, rather than reset, newborn age.
                        newborn_before = s["p"]["a"][-1].copy()
                        frontier.adam(s, g, lr, 100, mode)
                        self.assertEqual(s["age"]["a"][-1], 2)
                        np.testing.assert_allclose(s["m"]["a"][-1], 0.19)
                        np.testing.assert_allclose(s["v"]["a"][-1], 0.001999)
                        np.testing.assert_allclose(newborn_before - s["p"]["a"][-1],
                                                   lr / (1 + eps), rtol=1e-12)


class CausalityTests(unittest.TestCase):
    def test_later_labels_cannot_change_earlier_parameters_or_optimizer(self):
        cfg = frontier.default_config()
        cfg.update(d=4, arrivals=3, block=5, batch=4, max_width=3, neval=8)
        task, seed, budget = "indefinite_noisy", 45, 10000
        x, y, _ = frontier.generate(seed, task, cfg)
        mutated_y = y.copy()
        cutoff = 2 * cfg["block"]
        mutated_y[cutoff:] += 1000 + np.arange(len(y) - cutoff)
        for method in ("historical_moment", "random_growth128", "random_growth138", "fixed7"):
            for mode in ("global", "local"):
                with self.subTest(method=method, mode=mode):
                    one = frontier.learner(seed, task, method, mode, budget, cfg, x, y)
                    two = frontier.learner(seed, task, method, mode, budget, cfg, x, mutated_y)
                    self.assertEqual(len(one["events"]), cfg["arrivals"])
                    self.assertEqual(len(two["events"]), cfg["arrivals"])
                    self.assertEqual(one["initial_parameters"], two["initial_parameters"])
                    for ix in range(2):
                        for key in ("before_parameters", "after_parameters", "before_optimizer",
                                    "after_optimizer", "proposal", "retained_ids_after"):
                            self.assertEqual(one["events"][ix][key], two["events"][ix][key],
                                             msg=f"{method}/{mode} event {ix}: {key} depends on future labels")
                    for result in (one, two):
                        for event in result["events"]:
                            self.assertLessEqual(event["total_event_proxy"], budget)
                    # Ensure the changed labels do affect fitting once they arrive.
                    self.assertTrue(any(
                        not np.array_equal(one["events"][2]["after_parameters"][key],
                                           two["events"][2]["after_parameters"][key])
                        for key in ("w", "a", "c")))


class _OrderedRng:
    """A controlled epoch ordering whose calls are also observable."""
    def __init__(self):
        self.permutation_calls = 0

    def permutation(self, n):
        self.permutation_calls += 1
        return np.arange(n)


class FitBudgetTests(unittest.TestCase):
    def test_full_batches_carry_epoch_tails_and_budget_limits_terminal_batch(self):
        for pool_size in (3, 5):
            with self.subTest(pool_size=pool_size):
                cfg = frontier.default_config()
                cfg.update(d=4, batch=4, lr=0.01, gradient_clip=100)
                state = frontier.new_state(8, "null", 4, 1, "local")
                # Column zero is an unambiguous row identifier for patched grad.
                x = np.zeros((pool_size, 4))
                x[:, 0] = np.arange(pool_size)
                y = np.zeros(pool_size)
                ids = np.arange(pool_size) + 100
                expected_sizes = [4, 4, 2]
                expected_gradient_cost = sum(frontier.fit_batch_work(n, 4, 1)
                                             for n in expected_sizes)
                expected_epochs = (sum(expected_sizes) + pool_size - 1) // pool_size
                expected_permutation_cost = 4 * pool_size * expected_epochs
                budget = expected_gradient_cost + expected_permutation_cost + 7
                rng = _OrderedRng()
                batches = []

                def capture_grad(bx, by, p):
                    batches.append(bx[:, 0].astype(int).tolist())
                    return {key: np.zeros_like(value) for key, value in p.items()}

                with patch.object(frontier, "grad", side_effect=capture_grad):
                    account, counts = frontier.fit(state, x, y, ids, budget, rng, cfg, "local")
                self.assertEqual([len(batch) for batch in batches], expected_sizes)
                self.assertEqual([ix for batch in batches for ix in batch],
                                 (np.arange(10) % pool_size).tolist())
                np.testing.assert_array_equal(counts,
                                              np.bincount(np.arange(10) % pool_size,
                                                          minlength=pool_size))
                self.assertEqual(rng.permutation_calls, expected_epochs)
                self.assertEqual(state["step"], 3)
                self.assertEqual(account["steps"], 3)
                self.assertEqual(account["full_minibatches"], 2)
                self.assertEqual(account["terminal_budget_batch_size"], 2)
                self.assertEqual(account["min_batch_size"], 2)
                self.assertEqual(account["max_batch_size"], 4)
                self.assertEqual(account["total_fit_visits"], 10)
                self.assertEqual(account["gradient_adam_proxy"], expected_gradient_cost)
                self.assertEqual(account["permutation_proxy"], expected_permutation_cost)
                self.assertEqual(account["proxy"], budget - 7)
                self.assertEqual(account["remainder"], 7)
                self.assertLessEqual(account["proxy"], budget)

    def test_insufficient_budget_does_no_work_or_permutation(self):
        cfg = frontier.default_config()
        cfg.update(d=4, batch=4)
        s = frontier.new_state(8, "null", 4, 1, "local")
        before = copy.deepcopy(s)
        rng = _OrderedRng()
        budget = frontier.fit_batch_work(1, 4, 1) + 4 * 3 - 1
        with patch.object(frontier, "grad") as mocked_grad:
            account, counts = frontier.fit(s, np.zeros((3, 4)), np.zeros(3),
                                            np.arange(3), budget, rng, cfg, "local")
        mocked_grad.assert_not_called()
        self.assertEqual(rng.permutation_calls, 0)
        self.assertEqual(account["proxy"], 0)
        self.assertEqual(account["steps"], 0)
        self.assertEqual(account["remainder"], budget)
        np.testing.assert_array_equal(counts, np.zeros(3))
        self.assertEqual(s["step"], before["step"])
        for group in ("p", "m", "v", "age"):
            for key in before[group]:
                np.testing.assert_array_equal(s[group][key], before[group][key])


if __name__ == "__main__":
    unittest.main()
