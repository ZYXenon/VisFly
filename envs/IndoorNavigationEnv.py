import numpy as np
import torch as th
from gymnasium import spaces
from typing import Optional, Dict

from .base.droneGymEnv import DroneGymEnvsBase
from ..utils.type import TensorDict


class IndoorNavigationEnv(DroneGymEnvsBase):
    """
    Indoor point-goal navigation with per-episode random targets.
    Targets are sampled in collision-free space via rejection sampling.
    Collision does NOT terminate the episode (heavy penalty instead).
    """

    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = {},
            device: str = "cpu",
            max_episode_steps: int = 512,
            target_range: Optional[dict] = None,
            success_radius: float = 0.5,
    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            is_collision_reset=False,
        )

        # Target sampling range (ENU coords, covering playroom interior)
        if target_range is None:
            target_range = {"mean": [0.0, -1.0, 0.5], "half": [5.0, 5.5, 2.0]}
        self.target_mean = th.tensor(target_range["mean"], dtype=th.float32)
        self.target_half = th.tensor(target_range["half"], dtype=th.float32)

        self.target = th.zeros((self.num_envs, 3))
        self.success_radius = success_radius

        # Observation space: state(13) + depth(1,64,64) + target(3)
        self.observation_space["target"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    # ---- Target randomization ----

    def _sample_collision_free_points(self, num, scene_id):
        """Rejection-sample collision-free points within target_range for a given scene."""
        points = (2 * th.rand(num, 3) - 1) * self.target_half + self.target_mean
        if not self.visual:
            return points

        is_collision = self.envs.sceneManager.get_point_is_collision(
            std_positions=points, scene_id=scene_id, uav_radius=0.3
        )
        max_iter = 200
        for _ in range(max_iter):
            if not is_collision.any():
                break
            n_bad = is_collision.sum()
            points[is_collision] = (
                (2 * th.rand(n_bad, 3) - 1) * self.target_half + self.target_mean
            )
            is_collision = self.envs.sceneManager.get_point_is_collision(
                std_positions=points, scene_id=scene_id, uav_radius=0.3
            )
        return points

    def _randomize_targets(self, indices=None):
        """Sample new random targets for agents at `indices` (or all agents)."""
        if indices is None:
            indices = th.arange(self.num_envs)
        indices = th.atleast_1d(indices)
        if len(indices) == 0:
            return

        # Group agents by scene_id for batched collision checking
        for scene_id in range(self.num_scene):
            agent_start = scene_id * self.num_agent_per_scene
            agent_end = agent_start + self.num_agent_per_scene
            mask = (indices >= agent_start) & (indices < agent_end)
            scene_indices = indices[mask]
            if len(scene_indices) == 0:
                continue
            new_targets = self._sample_collision_free_points(
                len(scene_indices), scene_id
            )
            self.target[scene_indices] = new_targets

    # ---- Reset / examine overrides ----

    def reset(self, state=None, predicted_obs=None, is_test=False, stoch=None, deter=None):
        # Randomize targets for all agents before the base reset generates positions
        self._is_initial = True
        self.envs.reset(state=state)
        self._randomize_targets()

        if isinstance(self.get_reward(), dict):
            self._indiv_reward = self.get_reward()
            self._indiv_rewards = {key: th.zeros((self.num_agent,)) for key in self._indiv_reward.keys()}
            self._indiv_reward = {key: th.zeros((self.num_agent,)) for key in self._indiv_rewards.keys()}
        else:
            self._indiv_rewards = None
            self._indiv_reward = None

        self.get_full_observation(predicted_obs=predicted_obs)
        self._reset_attr(reset_latent=(stoch is None))
        self.get_full_observation(predicted_obs=predicted_obs)

        if stoch is not None:
            self.stoch = stoch
            self.deter = deter

        return self._observations

    def examine(self):
        if self._done.any():
            done_indices = th.where(self._done)[0]
            self._randomize_targets(done_indices)
        return super().examine()

    # ---- Observation ----

    def get_observation(self, indices=None) -> Dict:
        # Relative target in world frame (better generalization than absolute)
        relative_target = self.target - self.position

        obs = TensorDict({
            "state": self.state.to(self.device),
            "target": relative_target.to(self.device),
        })

        if self.visual:
            obs["depth"] = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
            if "color" in self.sensor_obs:
                obs["color"] = th.from_numpy(self.sensor_obs["color"]).to(self.device)

        return obs

    # ---- Success / Failure ----

    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius

    def get_failure(self) -> th.Tensor:
        return th.zeros(self.num_agent, dtype=th.bool)

    # ---- Reward ----

    def get_reward(self) -> dict:
        to_target = self.target - self.position
        dist = to_target.norm(dim=1)
        to_target_dir = to_target / (dist.unsqueeze(1) + 1e-6)

        # Approach reward: dot(velocity, target_direction)
        r_approach = (self.velocity * to_target_dir).sum(dim=1) * 0.02

        # Speed penalty (indoor = slower is safer)
        r_speed = -self.velocity.norm(dim=1) * 0.005

        # Angular velocity penalty (stability)
        r_omega = -self.angular_velocity.norm(dim=1) * 0.002

        # Proximity penalty (stay away from obstacles)
        r_proximity = -1.0 / (self.collision_dis + 0.2) * 0.02

        # Approaching-obstacle velocity penalty
        approach_obs_speed = (
            (self.collision_vector * self.velocity).sum(dim=1)
            / (self.collision_dis + 1e-6)
        ).relu()
        r_collision_v = -approach_obs_speed * (1 - self.collision_dis).relu() * 0.01

        # Hard collision penalty (heavier since collision no longer terminates)
        r_collision = self.is_collision.float() * -2.0

        # Survival reward (encourage staying alive and exploring)
        r_survival = th.ones(self.num_agent) * 0.01

        # Success bonus (proportional to remaining steps)
        r_success = self._success.float() * (
            (self.max_episode_steps - self._step_count).float() * 0.05 + 2.0
        )

        reward = (
            r_approach
            + r_speed
            + r_omega
            + r_proximity
            + r_collision_v
            + r_collision
            + r_survival
            + r_success
        )

        return {
            "reward": reward,
            "r_approach": r_approach.detach(),
            "r_speed": r_speed.detach(),
            "r_omega": r_omega.detach(),
            "r_proximity": r_proximity.detach(),
            "r_collision_v": r_collision_v.detach(),
            "r_collision": r_collision.detach(),
            "r_survival": r_survival.detach(),
            "r_success": r_success.detach(),
        }
