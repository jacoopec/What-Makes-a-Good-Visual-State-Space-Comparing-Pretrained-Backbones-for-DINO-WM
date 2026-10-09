import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from datasets.pusht_dset import PushTDataset
from datasets.img_transforms import default_transform
from models.encoder.mae import MAEEncoder


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TRAJECTORIES = 20
FRAME_SKIP = 5
BATCH_SIZE = 32
IMG_SIZE = 224

OUTPUT_DIR = Path("mae_geometry_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Feature encoding
# ============================================================

@torch.no_grad()
def encode_frames(encoder, images):
    """
    images:
        [T, 3, 224, 224]
        already in [-1, 1] because PushT default_transform does this

    returns:
        [T, 196, 768]
    """

    all_features = []

    for start in range(0, len(images), BATCH_SIZE):

        batch = images[start:start + BATCH_SIZE].to(DEVICE)

        z = encoder(batch)

        all_features.append(z.cpu())

    return torch.cat(all_features, dim=0)


# ============================================================
# Distances
# ============================================================

def mae_distance_to_goal(features):
    """
    features:
        [T, 196, 768]

    Use last frame as goal.
    """

    goal = features[-1]

    # MSE across patch and embedding dimensions
    distance = ((features - goal) ** 2).mean(dim=(1, 2))

    return distance.numpy()


def full_state_distance_to_goal(states):
    """
    Matches the basic state-distance calculation used by
    PushTWrapper.eval_state():

        ||goal_state - current_state||

    state:
    [agent_x, agent_y, T_x, T_y, angle, vx, vy]
    """

    goal = states[-1]

    return torch.linalg.vector_norm(
        states - goal,
        dim=-1
    ).numpy()


def position_distance_to_goal(states):
    """
    Position component used by PushT success condition.

    First four dimensions:
    [agent_x, agent_y, T_x, T_y]
    """

    positions = states[:, :4]

    goal = positions[-1]

    return torch.linalg.vector_norm(
        positions - goal,
        dim=-1
    ).numpy()


def object_xy_distance_to_goal(states):
    """
    Only the manipulated T object's XY position.

    state:
        0 = agent x
        1 = agent y
        2 = T x
        3 = T y
        4 = T angle
    """

    object_xy = states[:, 2:4]

    goal = object_xy[-1]

    return torch.linalg.vector_norm(
        object_xy - goal,
        dim=-1
    ).numpy()


# ============================================================
# Correlation helper
# ============================================================

def correlations(x, y):

    # Last point is always exactly zero for both because it is the goal.
    x = x[:-1]
    y = y[:-1]

    if len(x) < 3:
        return np.nan, np.nan

    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    return pearson, spearman


# ============================================================
# Plot
# ============================================================

def normalize_for_plot(x):

    x = np.asarray(x)

    maximum = np.max(x)

    if maximum < 1e-8:
        return x

    return x / maximum


def plot_trajectory(
    traj_idx,
    visual_dist,
    state_dist,
    position_dist,
    object_dist,
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        normalize_for_plot(visual_dist),
        label="MAE feature distance"
    )

    plt.plot(
        normalize_for_plot(state_dist),
        label="Full state distance"
    )

    plt.plot(
        normalize_for_plot(position_dist),
        label="Agent + object position distance"
    )

    plt.plot(
        normalize_for_plot(object_dist),
        label="T-object XY distance"
    )

    plt.xlabel("Trajectory timestep")
    plt.ylabel("Normalized distance to final state")

    plt.title(
        f"MAE representation geometry - PushT trajectory {traj_idx}"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    path = OUTPUT_DIR / f"trajectory_{traj_idx:02d}.png"

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# Main experiment
# ============================================================

def main():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_root = os.environ.get("DATASET_DIR")

    if dataset_root is None:
        raise RuntimeError(
            "DATASET_DIR is not defined.\n"
            "Set it to the same dataset root used by DINO-WM training."
        )

    pusht_path = (
        Path(dataset_root)
        / "pusht_noise"
        / "val"
    )

    print("=" * 70)
    print("Loading PushT validation data")
    print("Path:", pusht_path)
    print("=" * 70)

    dataset = PushTDataset(
        n_rollout=NUM_TRAJECTORIES,
        transform=default_transform(IMG_SIZE),
        data_path=str(pusht_path),
        normalize_action=True,
        with_velocity=True,
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    encoder = MAEEncoder(
        model_name="facebook/vit-mae-base",
        local_files_only=True,
    ).to(DEVICE)

    # Very important
    encoder.eval()
    encoder.base_model.eval()

    print()
    print("Device:", DEVICE)
    print("Encoder training mode:", encoder.training)
    print("MAE training mode:", encoder.base_model.training)
    print(
        "Trainable MAE parameters:",
        sum(
            p.requires_grad
            for p in encoder.base_model.parameters()
        )
    )

    assert not encoder.training
    assert not encoder.base_model.training

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    n_trajectories = min(
        NUM_TRAJECTORIES,
        len(dataset)
    )

    print()
    print(
        f"Running geometry experiment on "
        f"{n_trajectories} trajectories..."
    )

    print()

    for traj_idx in range(n_trajectories):

        obs, actions, states, info = dataset[traj_idx]

        images = obs["visual"]

        # Match DINO-WM's usual temporal spacing
        images = images[::FRAME_SKIP]
        states = states[::FRAME_SKIP]

        print("-" * 70)

        print(
            f"Trajectory {traj_idx:02d}: "
            f"{len(images)} sampled frames"
        )

        print(
            "Input image range:",
            images.min().item(),
            images.max().item()
        )

        # This MUST already be [-1,1].
        if images.min() < -1.01 or images.max() > 1.01:
            raise RuntimeError(
                "Unexpected image range. "
                "PushT default_transform should produce [-1,1]."
            )

        # ----------------------------------------------------
        # MAE encoding
        # ----------------------------------------------------

        features = encode_frames(
            encoder,
            images
        )

        visual_dist = mae_distance_to_goal(
            features
        )

        # ----------------------------------------------------
        # Physical distances
        # ----------------------------------------------------

        state_dist = full_state_distance_to_goal(
            states
        )

        position_dist = position_distance_to_goal(
            states
        )

        object_dist = object_xy_distance_to_goal(
            states
        )

        # ----------------------------------------------------
        # Correlations
        # ----------------------------------------------------

        p_state, s_state = correlations(
            visual_dist,
            state_dist
        )

        p_position, s_position = correlations(
            visual_dist,
            position_dist
        )

        p_object, s_object = correlations(
            visual_dist,
            object_dist
        )

        print(
            f"MAE vs FULL STATE : "
            f"Pearson={p_state:+.3f} "
            f"Spearman={s_state:+.3f}"
        )

        print(
            f"MAE vs POSITIONS  : "
            f"Pearson={p_position:+.3f} "
            f"Spearman={s_position:+.3f}"
        )

        print(
            f"MAE vs T OBJECT   : "
            f"Pearson={p_object:+.3f} "
            f"Spearman={s_object:+.3f}"
        )

        results.append(
            {
                "pearson_state": p_state,
                "spearman_state": s_state,

                "pearson_position": p_position,
                "spearman_position": s_position,

                "pearson_object": p_object,
                "spearman_object": s_object,
            }
        )

        # Save plots for first 5 trajectories
        if traj_idx < 5:

            plot_trajectory(
                traj_idx,
                visual_dist,
                state_dist,
                position_dist,
                object_dist,
            )

    # ========================================================
    # Aggregate statistics
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MAE GEOMETRY RESULTS")
    print("=" * 70)

    metrics = [
        "pearson_state",
        "spearman_state",

        "pearson_position",
        "spearman_position",

        "pearson_object",
        "spearman_object",
    ]

    for metric in metrics:

        values = np.array(
            [
                r[metric]
                for r in results
                if np.isfinite(r[metric])
            ]
        )

        print(
            f"{metric:22s}: "
            f"{values.mean():+.3f} "
            f"+/- {values.std():.3f}"
        )

    # Save numerical results
    output_file = OUTPUT_DIR / "results.txt"

    with open(output_file, "w") as f:

        for metric in metrics:

            values = np.array(
                [
                    r[metric]
                    for r in results
                    if np.isfinite(r[metric])
                ]
            )

            f.write(
                f"{metric}: "
                f"{values.mean():+.6f} "
                f"+/- {values.std():.6f}\n"
            )

    print()
    print(
        "Plots and numerical results saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()