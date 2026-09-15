import h5py
import numpy as np
import torch

from einops import rearrange
from typing import Callable, Optional

from .traj_dset import (
    TrajDataset,
    TrajSlicerDataset,
    split_traj_datasets,
)


class PushCubeDataset(TrajDataset):

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        filter_success: bool = True,
    ):
        self.data_path = data_path
        self.transform = transform
        self.normalize_action = normalize_action
        self.filter_success = filter_success

        # Open lazily inside each DataLoader worker
        self._h5 = None

        self.traj_names = []
        self.seq_lengths = []

        all_actions = []
        all_proprios = []
        all_states = []

        with h5py.File(self.data_path, "r") as f:

            traj_names = sorted(
                [k for k in f.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[-1]),
            )

            for traj_name in traj_names:

                traj = f[traj_name]

                # Keep successful demonstrations only
                if filter_success:
                    success = traj["success"][:]
                    if not np.any(success):
                        continue

                T = traj["actions"].shape[0]

                # Actions
                actions = torch.from_numpy(
                    traj["actions"][:]
                ).float()

                # Proprioception
                qpos = torch.from_numpy(
                    traj["obs"]["agent"]["qpos"][:T]
                ).float()

                qvel = torch.from_numpy(
                    traj["obs"]["agent"]["qvel"][:T]
                ).float()

                proprio = torch.cat(
                    [qpos, qvel],
                    dim=-1,
                )

                # --------------------------------------------------
                # Native ManiSkill simulator state
                # --------------------------------------------------

                panda = torch.from_numpy(
                    traj["env_states"]["articulations"]["panda"][:T]
                ).float()  # 31

                cube = torch.from_numpy(
                    traj["env_states"]["actors"]["cube"][:T]
                ).float()  # 13

                goal = torch.from_numpy(
                    traj["env_states"]["actors"]["goal_region"][:T]
                ).float()  # 13

                table = torch.from_numpy(
                    traj["env_states"]["actors"]["table-workspace"][:T]
                ).float()  # 13

                state = torch.cat(
                    [
                        panda,
                        cube,
                        goal,
                        table,
                    ],
                    dim=-1,
                )  # 70D

                self.traj_names.append(traj_name)
                self.seq_lengths.append(T)

                all_actions.append(actions)
                all_proprios.append(proprio)
                all_states.append(state)

                if (
                    n_rollout is not None
                    and len(self.traj_names) >= n_rollout
                ):
                    break

        if len(self.traj_names) == 0:
            raise ValueError(
                f"No valid PushCube trajectories found in {data_path}"
            )

        # ----------------------------------------------------------
        # Dimensions
        # ----------------------------------------------------------

        self.action_dim = all_actions[0].shape[-1]
        self.proprio_dim = all_proprios[0].shape[-1]
        self.state_dim = all_states[0].shape[-1]

        # ----------------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------------

        all_actions = torch.cat(all_actions, dim=0)
        all_proprios = torch.cat(all_proprios, dim=0)
        all_states = torch.cat(all_states, dim=0)

        if normalize_action:

            self.action_mean = all_actions.mean(dim=0)
            self.action_std = all_actions.std(
                dim=0,
                unbiased=False,
            ).clamp_min(1e-6)

            self.proprio_mean = all_proprios.mean(dim=0)
            self.proprio_std = all_proprios.std(
                dim=0,
                unbiased=False,
            ).clamp_min(1e-6)

            self.state_mean = all_states.mean(dim=0)
            self.state_std = all_states.std(
                dim=0,
                unbiased=False,
            ).clamp_min(1e-6)

        else:

            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)

            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)

        print(f"Loaded {len(self.traj_names)} PushCube rollouts")
        print(f"action_dim:  {self.action_dim}")
        print(f"proprio_dim: {self.proprio_dim}")
        print(f"state_dim:   {self.state_dim}")

    # --------------------------------------------------------------
    # HDF5 handling
    # --------------------------------------------------------------

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        return self._h5

    def __getstate__(self):
        """
        Prevent sharing an HDF5 handle between DataLoader workers.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    # --------------------------------------------------------------
    # Trajectory interface required by DINO-WM
    # --------------------------------------------------------------

    def __len__(self):
        return len(self.traj_names)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):

        f = self._get_h5()

        traj_name = self.traj_names[idx]
        traj = f[traj_name]

        frames = np.asarray(
            list(frames),
            dtype=np.int64,
        )

        # ----------------------------------------------------------
        # RGB
        # ----------------------------------------------------------

        rgb = torch.from_numpy(
            traj[
                "obs"
            ][
                "sensor_data"
            ][
                "base_camera"
            ][
                "rgb"
            ][frames]
        )

        # THWC -> TCHW
        rgb = rearrange(
            rgb,
            "t h w c -> t c h w",
        ).float() / 255.0

        if self.transform is not None:
            rgb = self.transform(rgb)

        # ----------------------------------------------------------
        # Proprioception
        # ----------------------------------------------------------

        qpos = torch.from_numpy(
            traj["obs"]["agent"]["qpos"][frames]
        ).float()

        qvel = torch.from_numpy(
            traj["obs"]["agent"]["qvel"][frames]
        ).float()

        proprio = torch.cat(
            [qpos, qvel],
            dim=-1,
        )

        proprio = (
            proprio - self.proprio_mean
        ) / self.proprio_std

        # ----------------------------------------------------------
        # Actions
        # ----------------------------------------------------------

        actions = torch.from_numpy(
            traj["actions"][frames]
        ).float()

        actions = (
            actions - self.action_mean
        ) / self.action_std

        # ----------------------------------------------------------
        # Native ManiSkill state: 70D
        # ----------------------------------------------------------

        panda = torch.from_numpy(
            traj[
                "env_states"
            ][
                "articulations"
            ][
                "panda"
            ][frames]
        ).float()

        cube = torch.from_numpy(
            traj[
                "env_states"
            ][
                "actors"
            ][
                "cube"
            ][frames]
        ).float()

        goal = torch.from_numpy(
            traj[
                "env_states"
            ][
                "actors"
            ][
                "goal_region"
            ][frames]
        ).float()

        table = torch.from_numpy(
            traj[
                "env_states"
            ][
                "actors"
            ][
                "table-workspace"
            ][frames]
        ).float()

        state = torch.cat(
            [
                panda,
                cube,
                goal,
                table,
            ],
            dim=-1,
        )

        obs = {
            "visual": rgb,
            "proprio": proprio,
        }

        info = {
            "traj_name": traj_name,
        }

        return obs, actions, state, info

    def __getitem__(self, idx):

        return self.get_frames(
            idx,
            range(self.get_seq_length(idx)),
        )

    def get_all_actions(self):

        actions = []

        for idx in range(len(self)):
            _, act, _, _ = self[idx]
            actions.append(act)

        return torch.cat(actions, dim=0)


# ==================================================================
# Train / validation loader
# ==================================================================

def load_pushcube_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    filter_success=True,
    num_hist=3,
    num_pred=1,
    frameskip=1,
):

    dataset = PushCubeDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        filter_success=filter_success,
    )

    if len(dataset) < 2:
        raise ValueError(
            "PushCube training requires at least 2 rollouts "
            "to create train and validation sets."
        )

    train_traj, val_traj = split_traj_datasets(
        dataset,
        train_fraction=split_ratio,
        random_seed=42,
    )

    num_frames = num_hist + num_pred

    train_dataset = TrajSlicerDataset(
        train_traj,
        num_frames=num_frames,
        frameskip=frameskip,
    )

    val_dataset = TrajSlicerDataset(
        val_traj,
        num_frames=num_frames,
        frameskip=frameskip,
    )

    datasets = {
        "train": train_dataset,
        "valid": val_dataset,
    }

    traj_datasets = {
        "train": train_traj,
        "valid": val_traj,
    }

    return datasets, traj_datasets