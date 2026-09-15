import gymnasium as gym
import mani_skill.envs


class PushCubeDinoEnv:

    def __init__(self, **kwargs):

        self.env = gym.make(
            "PushCube-v1",
            obs_mode="rgb",
            control_mode="pd_ee_delta_pos",
            num_envs=1,
            **kwargs
        )

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, **kwargs):
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()