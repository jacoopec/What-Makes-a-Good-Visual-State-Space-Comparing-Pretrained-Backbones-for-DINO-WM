import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # registers PushCube-v1


class PushCubeWrapper:
    """
    Adapter between DINO-WM planning and ManiSkill PushCube-v1.

    Dataset state layout:
        [0:31]   panda articulation state
        [31:44]  cube actor state
        [44:57]  goal-region actor state
        [57:70]  table actor state
    """

    PANDA = slice(0, 31)
    CUBE = slice(31, 44)
    GOAL = slice(44, 57)
    TABLE = slice(57, 70)

    STATE_DIM = 70

    def __init__(
        self,
        control_mode="pd_ee_delta_pos",
        obs_mode="rgb",
        sim_backend="physx_cpu",
        goal_tolerance=0.03,
    ):
        self.goal_tolerance = goal_tolerance

        self.env = gym.make(
            "PushCube-v1",
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend=sim_backend,
            render_mode=None,
        )

        self.base_env = self.env.unwrapped

        # ManiSkill is vectorized internally even with num_envs=1.
        self.action_space = self.env.get_wrapper_attr("single_action_space")
        self.action_dim = self.action_space.shape[0]

        print(
            f"PushCubeWrapper initialized: "
            f"action_dim={self.action_dim}, "
            f"control_mode={control_mode}"
        )

    # ---------------------------------------------------------
    # Conversion utilities
    # ---------------------------------------------------------

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

        return np.asarray(x)

    @classmethod
    def _remove_batch_dim(cls, x):
        x = cls._to_numpy(x)

        if x.ndim > 0 and x.shape[0] == 1:
            return x[0]

        return x

    @staticmethod
    def _value_like(value, reference):
        """
        Convert numpy state back to the type/device expected
        by ManiSkill's state dictionary.
        """
        value = np.asarray(value, dtype=np.float32).reshape(1, -1)

        if isinstance(reference, torch.Tensor):
            return torch.as_tensor(
                value,
                dtype=reference.dtype,
                device=reference.device,
            )

        return value.astype(reference.dtype, copy=False)

    # ---------------------------------------------------------
    # Observation conversion
    # ---------------------------------------------------------

    def _convert_obs(self, raw_obs):
        """
        ManiSkill:
            RGB:   [1,H,W,C]
            qpos:  [1,9]
            qvel:  [1,9]

        DINO-WM expects:
            visual:  [H,W,C], uint8
            proprio: [18]
        """

        rgb = self._remove_batch_dim(
            raw_obs["sensor_data"]["base_camera"]["rgb"]
        ).astype(np.uint8)

        qpos = self._remove_batch_dim(
            raw_obs["agent"]["qpos"]
        ).astype(np.float32)

        qvel = self._remove_batch_dim(
            raw_obs["agent"]["qvel"]
        ).astype(np.float32)

        proprio = np.concatenate(
            [qpos, qvel],
            axis=-1,
        ).astype(np.float32)

        return {
            "visual": rgb,
            "proprio": proprio,
        }

    # ---------------------------------------------------------
    # Simulator state
    # ---------------------------------------------------------

    def _get_state(self):
        """
        Return exactly the same 70D representation used
        by PushCubeDataset.
        """

        state_dict = self.base_env.get_state_dict()

        panda = self._remove_batch_dim(
            state_dict["articulations"]["panda"]
        )

        cube = self._remove_batch_dim(
            state_dict["actors"]["cube"]
        )

        goal = self._remove_batch_dim(
            state_dict["actors"]["goal_region"]
        )

        table = self._remove_batch_dim(
            state_dict["actors"]["table-workspace"]
        )

        return np.concatenate(
            [
                panda,
                cube,
                goal,
                table,
            ],
            axis=-1,
        ).astype(np.float32)

    def _build_env_state(self, state):
        """
        Convert our flat 70D state back into a ManiSkill
        state dictionary.
        """

        state = np.asarray(state, dtype=np.float32)

        if state.shape[-1] != self.STATE_DIM:
            raise ValueError(
                f"Expected PushCube state_dim={self.STATE_DIM}, "
                f"got {state.shape}"
            )

        # Gives us a correctly structured dictionary with
        # tensors on the correct ManiSkill device.
        env_state = self.base_env.get_state_dict()

        env_state["articulations"]["panda"] = self._value_like(
            state[self.PANDA],
            env_state["articulations"]["panda"],
        )

        env_state["actors"]["cube"] = self._value_like(
            state[self.CUBE],
            env_state["actors"]["cube"],
        )

        env_state["actors"]["goal_region"] = self._value_like(
            state[self.GOAL],
            env_state["actors"]["goal_region"],
        )

        env_state["actors"]["table-workspace"] = self._value_like(
            state[self.TABLE],
            env_state["actors"]["table-workspace"],
        )

        return env_state

    # ---------------------------------------------------------
    # DINO-WM API
    # ---------------------------------------------------------

    def prepare(self, seed, init_state):
        """
        Reset ManiSkill exactly to the state sampled by DINO-WM
        from the PushCube dataset.
        """

        seed = int(seed)

        # Initialize the scene/state-dict structure.
        self.env.reset(seed=seed)

        env_state = self._build_env_state(init_state)

        # ManiSkill officially supports resetting directly
        # to recorded environment states.
        raw_obs, _ = self.env.reset(
            seed=seed,
            options={
                "reset_to_env_states": {
                    "env_states": env_state
                }
            },
        )

        obs = self._convert_obs(raw_obs)
        state = self._get_state()

        return obs, state

    def _step_maniskill(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, "
                f"got {action.shape}"
            )

        # CEM can occasionally propose values outside the
        # controller's valid action bounds.
        low = np.asarray(
            self.action_space.low
        ).reshape(-1)

        high = np.asarray(
            self.action_space.high
        ).reshape(-1)

        action = np.clip(
            action,
            low,
            high,
        )

        # ManiSkill expects [num_envs, action_dim].
        batched_action = action[None, :]

        return self.env.step(batched_action)

    def rollout(self, seed, init_state, actions):
        """
        DINO-WM interface.

        Input:
            actions: [T, action_dim]

        Output:
            obses["visual"]:  [T+1,H,W,C]
            obses["proprio"]: [T+1,18]
            states:           [T+1,70]
        """

        obs_0, state_0 = self.prepare(
            seed,
            init_state,
        )

        visuals = [obs_0["visual"]]
        proprios = [obs_0["proprio"]]
        states = [state_0]

        for action in actions:
            raw_obs, reward, terminated, truncated, info = (
                self._step_maniskill(action)
            )

            obs = self._convert_obs(raw_obs)
            state = self._get_state()

            visuals.append(obs["visual"])
            proprios.append(obs["proprio"])
            states.append(state)

        obses = {
            "visual": np.stack(visuals),
            "proprio": np.stack(proprios),
        }

        states = np.stack(states)

        return obses, states

    def step_multiple(self, actions):
        """
        Included for compatibility with DINO-WM's other
        environment wrappers.
        """

        visuals = []
        proprios = []
        rewards = []
        dones = []
        states = []

        for action in actions:
            raw_obs, reward, terminated, truncated, info = (
                self._step_maniskill(action)
            )

            obs = self._convert_obs(raw_obs)
            state = self._get_state()

            visuals.append(obs["visual"])
            proprios.append(obs["proprio"])
            rewards.append(
                float(
                    self._remove_batch_dim(reward)
                )
            )

            terminated = bool(
                self._remove_batch_dim(terminated)
            )

            truncated = bool(
                self._remove_batch_dim(truncated)
            )

            dones.append(
                terminated or truncated
            )

            states.append(state)

        obses = {
            "visual": np.stack(visuals),
            "proprio": np.stack(proprios),
        }

        infos = {
            "state": np.stack(states)
        }

        return (
            obses,
            np.asarray(rewards),
            np.asarray(dones),
            infos,
        )

    def update_env(self, env_info):
        """
        PushCube has fixed geometry.
        Nothing needs to be changed between dataset episodes.
        """
        return None

    def eval_state(self, goal_state, cur_state):
        """
        Evaluate image-goal planning.

        We measure whether the cube reached the cube position
        of the desired goal state.
        """

        goal_state = np.asarray(goal_state)
        cur_state = np.asarray(cur_state)

        goal_cube = goal_state[self.CUBE]
        cur_cube = cur_state[self.CUBE]

        cur_goal_region = cur_state[self.GOAL]

        cube_goal_dist = np.linalg.norm(
            goal_cube[:2] - cur_cube[:2]
        )

        # Distance to the native ManiSkill PushCube target.
        task_goal_dist = np.linalg.norm(
            cur_cube[:2] - cur_goal_region[:2]
        )

        # Goal-conditioned DINO-WM success.
        success = cube_goal_dist < self.goal_tolerance

        # Native PushCube success criterion is radius 0.1
        # plus cube remaining on the table.
        task_success = (
            task_goal_dist < 0.1
            and cur_cube[2] < 0.025
        )

        return {
            "success": np.bool_(success),
            "cube_goal_dist": np.float32(
                cube_goal_dist
            ),
            "task_success": np.bool_(
                task_success
            ),
            "task_goal_dist": np.float32(
                task_goal_dist
            ),
        }

    def sample_random_init_goal_states(self, seed):
        raise NotImplementedError(
            "For PushCube use goal_source=dset. "
            "random_state would also randomize the physical "
            "goal marker and is not implemented yet."
        )

    def close(self):
        self.env.close()