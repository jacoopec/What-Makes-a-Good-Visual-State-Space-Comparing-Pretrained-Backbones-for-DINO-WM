import h5py
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset, split_traj_datasets


class PushCubeDataset(TrajDataset):

    def __init__(
        self,
        data_path,
        transform=None,
        n_rollout=None,
        normalize_action=True,
        filter_success=True,
    ):

        self.visuals = []
        self.proprios = []
        self.actions = []
        self.states = []
        self.seq_lengths = []

        with h5py.File(data_path, "r") as f:

            traj_names = sorted(
                [k for k in f.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[-1])
            )

            if n_rollout is not None:
                traj_names = traj_names[:n_rollout]

            for traj_name in traj_names:

                traj = f[traj_name]

                if filter_success and not bool(traj["success"][-1]):
                    continue

                T = traj["actions"].shape[0]

                actions = torch.from_numpy(
                    traj["actions"][:]
                ).float()

                rgb = torch.from_numpy(
                    traj["obs"]["sensor_data"]["base_camera"]["rgb"][:T]
                )

                rgb = rearrange(
                    rgb,
                    "t h w c -> t c h w"
                ).float() / 255.0

                qpos = torch.from_numpy(
                    traj["obs"]["agent"]["qpos"][:T]
                ).float()

                qvel = torch.from_numpy(
                    traj["obs"]["agent"]["qvel"][:T]
                ).float()

                tcp = torch.from_numpy(
                    traj["obs"]["extra"]["tcp_pose"][:T]
                ).float()

                cube = torch.from_numpy(
                    traj["env_states"]["actors"]["cube"][:T]
                ).float()

                goal = torch.from_numpy(
                    traj["env_states"]["actors"]["goal_region"][:T]
                ).float()

                proprio = torch.cat([qpos, qvel], dim=-1)   # [T, 18]
                state = torch.cat([qpos, qvel, tcp, cube, goal], dim=-1)  # [T, 51]

                if transform is not None:
                    rgb = torch.stack([transform(img) for img in rgb], dim=0)

                self.visuals.append(rgb)
                self.proprios.append(proprio)
                self.actions.append(actions)
                self.states.append(state)
                self.seq_lengths.append(T)



        if len(self.actions) == 0:
            raise ValueError("No valid trajectories loaded.")

        self.action_dim = self.actions[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]

        all_actions = torch.cat(self.actions, dim=0)
        all_proprios = torch.cat(self.proprios, dim=0)
        all_states = torch.cat(self.states, dim=0)

        self.action_mean = all_actions.mean(dim=0)
        self.action_std = all_actions.std(dim=0).clamp_min(1e-6)

        self.proprio_mean = all_proprios.mean(dim=0)
        self.proprio_std = all_proprios.std(dim=0).clamp_min(1e-6)

        self.state_mean = all_states.mean(dim=0)
        self.state_std = all_states.std(dim=0).clamp_min(1e-6)

        if normalize_action:
            self.actions = [(x - self.action_mean) / self.action_std for x in self.actions]
            self.proprios = [(x - self.proprio_mean) / self.proprio_std for x in self.proprios]
            # se vuoi, puoi anche normalizzare state nel preprocessor di DINO-WM

        print(f"Loaded {len(self.seq_lengths)} PushCube rollouts")
        print("action_dim:", self.action_dim)
        print("proprio_dim:", self.proprio_dim)
        print("state_dim:", self.state_dim)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        frames = torch.tensor(list(frames), dtype=torch.long)

        obs = {
            "visual": self.visuals[idx][frames],
            "proprio": self.proprios[idx][frames],
        }

        act = self.actions[idx][frames]
        state = self.states[idx][frames]

        return obs, act, state, {}


    def __getitem__(self, idx):
        return self.get_frames(
            idx,
            range(self.get_seq_length(idx))
        )

    def __len__(self):
        return len(self.seq_lengths)


def load_pushcube_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=3,
    num_pred=1,
    frameskip=1,
    filter_success=True,
):
    dataset = PushCubeDataset(
        data_path=data_path,
        transform=transform,
        n_rollout=n_rollout,
        normalize_action=normalize_action,
        filter_success=filter_success,
    )

    train_traj, val_traj = split_traj_datasets(
        dataset,
        train_fraction=split_ratio,
        random_seed=42,
    )

    num_frames = num_hist + num_pred

    train_dataset = TrajSlicerDataset(train_traj, num_frames, frameskip)
    val_dataset = TrajSlicerDataset(val_traj, num_frames, frameskip)

    datasets = {
        "train": train_dataset,
        "valid": val_dataset,
    }

    traj_datasets = {
        "train": train_traj,
        "valid": val_traj,
    }

    return datasets, traj_datasets