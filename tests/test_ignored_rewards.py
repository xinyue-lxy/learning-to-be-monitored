"""Unmeasured monitorability responses must not train the policy."""

from unittest.mock import Mock, patch

import numpy as np
import unittest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage


def test_ignored_reward_cannot_change_valid_group_advantages(normalize):
    expected, _ = compute_grpo_outcome_advantage(
        torch.tensor([[4.0], [-2.0]]),
        torch.ones(2, 1),
        np.array(["q", "q"]),
        norm_adv_by_std_in_grpo=normalize,
    )
    for placeholder in (0.0, -4.0, 1000.0):
        actual, _ = compute_grpo_outcome_advantage(
            torch.tensor([[4.0], [placeholder], [-2.0]]),
            torch.ones(3, 1),
            np.array(["q"] * 3),
            sample_mask=torch.tensor([True, False, True]),
            norm_adv_by_std_in_grpo=normalize,
        )
        torch.testing.assert_close(actual[[0, 2]], expected)
        assert actual[1].item() == 0.0


def test_singleton_and_fully_ignored_groups_have_no_reward_advantage():
    values, _ = compute_grpo_outcome_advantage(
        torch.tensor([[4.0], [0.0], [0.0]]),
        torch.ones(3, 1),
        np.array(["q", "q", "r"]),
        sample_mask=torch.tensor([True, False, False]),
    )
    assert torch.isfinite(values).all()
    assert torch.count_nonzero(values) == 0


def test_trainer_passes_reward_valid_to_grpo():
    batch = DataProto.from_dict(
        tensors={"token_level_rewards": torch.tensor([[4.0], [0.0], [4.0]]), "response_mask": torch.ones(3, 1)},
        non_tensors={"uid": np.array(["q"] * 3), "reward_valid": np.array([True, False, True])},
    )
    result = ray_trainer.compute_advantage(batch, AdvantageEstimator.GRPO)
    assert torch.count_nonzero(result.batch["advantages"]) == 0


def test_actor_drops_unmeasured_rows_before_building_losses():
    trainer = object.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"multi_turn": {"enable": False}, "temperature": 1.0}}}
    )
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[1], [2], [3]])},
        non_tensors={"reward_valid": np.array([True, False, True])},
    )

    class ReachedLossBoundary(Exception):
        pass

    def inspect_actor_input(data):
        assert data.batch_size[0] == 2
        assert data["prompts"].flatten().tolist() == [1, 3]
        raise ReachedLossBoundary

    with patch.object(ray_trainer, "left_right_2_no_padding", side_effect=inspect_actor_input):
        with unittest.TestCase().assertRaises(ReachedLossBoundary):
            trainer._update_actor(batch)


def test_all_ignored_batch_does_not_call_optimizer():
    trainer = object.__new__(ray_trainer.RayPPOTrainer)
    trainer.actor_rollout_wg = Mock()
    batch = DataProto.from_dict(
        tensors={"prompts": torch.zeros(4, 1)},
        non_tensors={"reward_valid": np.zeros(4, dtype=bool)},
    )
    result = trainer._update_actor(batch)
    assert result.meta_info["metrics"]["actor/skipped_no_valid_rewards"] == 1.0
    trainer.actor_rollout_wg.update_actor.assert_not_called()


def test_odd_valid_batch_reaches_actor_without_padding_or_dropped_samples():
    trainer = object.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"multi_turn": {"enable": False}, "temperature": 1.0, "n": 4},
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 8,
                    "ppo_epochs": 1,
                    "data_loader_seed": 42,
                    "shuffle": False,
                },
            },
        }
    )
    valid = np.ones(64, dtype=bool)
    valid[17] = False
    batch = DataProto.from_dict(
        tensors={"prompts": torch.arange(64).reshape(-1, 1)},
        non_tensors={"reward_valid": valid},
    )

    def update(data):
        mini_batch = ray_trainer.tu.get(data, "mini_batch_size")
        assert mini_batch == 63
        assert data["prompts"].flatten().tolist() == [i for i in range(64) if i != 17]
        batches = list(ray_trainer.tu.make_iterator(data, mini_batch, 1))
        assert len(batches) == 1 and batches[0].batch_size[0] == 63
        result = TensorDict({}, batch_size=[])
        ray_trainer.tu.assign_non_tensor(result, metrics={"mfu": 0.0})
        return result

    trainer.actor_rollout_wg = Mock()
    trainer.actor_rollout_wg.update_actor.side_effect = update
    with patch.object(ray_trainer, "left_right_2_no_padding", side_effect=lambda data: data):
        result = trainer._update_actor(batch)
    assert "perf/mfu/actor" in result.meta_info["metrics"]
    trainer.actor_rollout_wg.update_actor.assert_called_once()


class IgnoredRewardTests(unittest.TestCase):
    def test_normalization(self):
        for normalize in (True, False):
            with self.subTest(normalize=normalize):
                test_ignored_reward_cannot_change_valid_group_advantages(normalize)

    def test_singleton_and_fully_ignored_groups_have_no_reward_advantage(self):
        test_singleton_and_fully_ignored_groups_have_no_reward_advantage()

    def test_trainer_passes_reward_valid_to_grpo(self):
        test_trainer_passes_reward_valid_to_grpo()

    def test_actor_drops_unmeasured_rows_before_building_losses(self):
        test_actor_drops_unmeasured_rows_before_building_losses()

    def test_all_ignored_batch_does_not_call_optimizer(self):
        test_all_ignored_batch_does_not_call_optimizer()

    def test_odd_valid_batch_reaches_actor_without_padding_or_dropped_samples(self):
        test_odd_valid_batch_reaches_actor_without_padding_or_dropped_samples()
