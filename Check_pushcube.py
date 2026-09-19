
from datasets.pushcube_dset import PushCubeDataset

path = "/work/cvcs2026/DINO_WM/data/pushcube/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"

d = PushCubeDataset(
    data_path=path,
    n_rollout=1,
    transform=None,
)

print("dataset length:", len(d))

x = d[0]

print("returned:", type(x))
print("number elements:", len(x) if x is not None else None)

if x is not None:
    obs, act, state, info = x
    print("visual:", obs["visual"].shape)
    print("proprio:", obs["proprio"].shape)
    print("actions:", act.shape)
    print("state:", state.shape)
PY
