# Spatial similarity and temporal correspondence

`spatial_correspondence.py` compares **frozen pretrained patch features**, without
loading a world model, predictor, decoder, planner, or training checkpoint. It uses
the encoder classes selected by the repository's actual `conf/encoder/*.yaml` files.
Training and planning code are unchanged.

## Repository inspection and feature contract

There is no shared encoder base class here. The spatial wrappers expose
`patch_size`, `emb_dim`, `latent_ndim`, and return `[B,N,D]`. The inspected defaults are:

| Config | Existing implementation | Analysis input | Spatial grid | Feature dimension | Preprocessing / feature stage |
| --- | --- | --- | --- | --- | --- |
| `dino` | `models.dino.DinoV2Encoder` | 224 x 224 | 16 x 16 | 384 | ImageNet normalization; `x_norm_patchtokens` |
| `clip_encoder` | `models.encoder.clip_encoder.CLIPVisionEncoder` | 224 x 224 | 14 x 14 | 512 | Wrapper applies CLIP mean/std and removes CLS; default pretrained CLIP projection retained |
| `siglip` | `models.encoder.siglip.SigLIPEncoder` | 224 x 224 | 14 x 14 | 768 | Mean/std 0.5; no CLS |
| `mae` | `models.encoder.mae.MAEEncoder` | 224 x 224 | 14 x 14 | 768 | Wrapper's ImageNet buffers; underlying pretrained backbone before random 768-to-384 projection; CLS removed |
| `ijepa` | `models.encoder.ijepa.IJEPAEncoder` | 196 x 196 | 14 x 14 | 1280 | Mean/std 0.5; wrapper interpolates positional embeddings; no CLS |

These are expected defaults, not hardcoded output sizes. Every forward validates
`N = (input_height / patch_height) * (input_width / patch_width)` and the model's
feature dimension. Unexpected special tokens cause an error rather than guessing
the grid from a square root. `dino_cls`, `resnet`, and `r3m` expose global features;
`dummy` is not pretrained. They are deliberately rejected for spatial analysis.

The training transform (`datasets/img_transforms.py`) resizes the shorter side,
center-crops, and normalizes to [-1,1]. The DINO wrapper does not convert that generic
normalization to pretrained DINO normalization. This analysis explicitly supplies
ImageNet normalization to DINO. It supplies [-1,1] to the CLIP, SigLIP and I-JEPA
wrappers according to their current forward methods. MAE reuses the wrapper's loaded
`base_model` and normalization buffers, disables masking through that wrapper, and
passes increasing noise to keep all tokens in raster order. Its randomly initialized
`proj` is never called. CLIP's projection, unlike MAE's, comes from pretrained weights.

Normalization references: [DINOv2 transforms](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py),
[MAE processor](https://huggingface.co/facebook/vit-mae-base/blob/main/preprocessor_config.json),
[SigLIP processor](https://huggingface.co/google/siglip-base-patch16-224/blob/main/preprocessor_config.json),
[I-JEPA processor](https://huggingface.co/facebook/ijepa_vith14_1k/blob/main/preprocessor_config.json).

For spatial coverage, the analysis uses a **full-frame resize without cropping** to
each encoder's configured resolution: PIL bilinear for MAE/I-JEPA, bicubic for the
others. This differs from training's center crop for nonsquare observations, and
stretches nonsquare images. The original image is preserved and all reported
coordinates map through this explicit resize. I-JEPA keeps the repository's 196-pixel
default; `--encoder-override ijepa:input_size=224` evaluates its native 16 x 16 grid.
The chosen policy, dimensions, resolved Hydra config, HF model config when available,
normalization, reference bounds, software versions and RGB hashes are recorded.

## Run

Run from this repository root in an environment that can import the selected
existing encoder. The existing `environment.yaml` is stale: its Transformers 4.21.1
pin predates the current SigLIP/I-JEPA classes, and it does not install OpenAI's `clip`
package. Use the working environment for the current wrappers; for all five check:

```bash
python -c 'from transformers import ViTMAEModel, SiglipVisionModel, IJepaModel; import clip, hydra, decord'
```

PushT uses `PushTDataset(transform=None, normalize_action=False,
with_velocity=False).get_frames(trajectory, frames)`, retaining original RGB.
It loads state/action metadata on CPU and decodes only selected video frames.
Pass a **split directory**, as constructed by `conf/env/pusht.yaml`, not the dataset
root. Frame indices are original trajectory indices, not training slice indices.
There is no automatic frameskip. The first selected frame supplies the reference.

```bash
python spatial_correspondence.py \
  --source-type pusht --source "$DATASETDIR/pusht_noise/val" \
  --trajectory 0 --frames 0 5 10 20 --reference 112 112 \
  --encoders dino clip_encoder siglip mae ijepa \
  --device cuda --output correspondence_results/rollout_000
```

Choose `(x,y)` on the object in the **original** observation, not a resized 224-pixel
view. The example coordinate is illustrative, not a known object annotation.
Coordinates use image-edge axes: top-left edge `(0,0)`, pixel centers `(x+.5,y+.5)`.
Patch assignment is floor division after scaling to the grid. The predicted point
is the winning patch cell's center in that frame's original resolution.

Single-image spatial analysis (the repository includes this observation):

```bash
python spatial_correspondence.py \
  --source-type images --source input_image.png --frames 0 \
  --reference 112 112 --encoders dino --device cpu \
  --output correspondence_results/single_image
```

For an image directory, `--frames` indexes naturally sorted filenames (`frame2`
before `frame10`). Images may have different dimensions. A single image only has
frame 0. For a video use `--source-type video --source rollout.mp4`; OpenCV decodes
the requested frames to RGB. These inputs also support recordings from other
environments. Inspected PointMaze/Wall datasets expose similar `get_frames` methods
with different serialized image layouts; PushCube reads HDF5, while deformable
environments use their own rollout interface. This script directly integrates only
PushT and accepts image/video exports for those other environments.

Config names are exactly `dino`, `clip_encoder`, `siglip`, `mae`, `ijepa`, or paths
to YAML files targeting these existing classes. Each config's stem is its output
label. Duplicate labels, global encoders, invalid coordinates, duplicate/unsorted
frames and nonempty output directories are rejected. No backbone substitution occurs.

Overrides apply to just the named encoder, for example:

```bash
--encoder-override mae:local_files_only=true \
--encoder-override ijepa:cache_dir=/work/my_cache/huggingface \
--encoder-override clip_encoder:download_root=/work/my_cache/clip \
--encoder-override clip_encoder:return_projected_tokens=false
```

Several existing defaults point at `/homes/jpecchini/.cache`. Change these explicitly
when using a different account. DINO uses the repository-pinned Torch Hub revision
`b48308a` and `TORCH_HOME`. Hugging Face models honor their wrapper's cache and
`local_files_only` settings (MAE currently defaults to downloads allowed). Populate
all selected pretrained caches before running on offline compute nodes. Missing
weights or incompatible dependencies fail with an error; the script never silently
uses random weights or falls back to a different backbone.

## Optional ground truth

### Automatic PushT object masks

The script now segments the **visible gray T**, excluding the green target, blue
agent and light-gray arena walls. With `--source-type pusht` this is enabled by
default (`--object-mask auto`). For PushT recordings supplied as images or video,
add `--object-mask pusht-color`. Disable automatic segmentation with
`--object-mask none`. Explicit JSON masks/boxes always take precedence.

The detector uses the renderer's `LightSlateGray` RGB (119,136,153), its 1.2-times
lighter fill, and channel ordering to distinguish gray-blue object pixels from
green/blue/neutral regions. `--mask-color-tolerance 32` is the default Euclidean
RGB tolerance for compression and antialiasing. It keeps the largest connected
component, rejects tiny detections and ambiguous multiple large components, and
does not fill holes across an occluding agent. This is color-derived pseudo ground
truth for the default renderer, not a general T-shape recognizer or a perfect
geometric mask: changed colors, severe compression and occlusions can affect it.
Inspect the exported masks before using them for reported results.

New outputs:

* `object_masks/frame_*.png`: binary masks at original image resolution.
* `object_masks/frame_*_overlay.png`: original image, binary mask and colored overlay.
* `metadata.json` / `object_masks`: per-frame method, detection status, pixel count
  and mask path. Missing/ambiguous automatic detections leave localization and
  contrast blank, rather than treating an invalid mask as an incorrect localization.
* CSV `object_localization_hit`: 1 if the maximum-similarity patch center lies
  inside the T, 0 otherwise. `region_type=pusht_color_mask` identifies automatic masks.
* CSV `mean_object_similarity`, `mean_background_similarity`, and
  `object_background_contrast`: object mean minus background mean using the original
  resolution, bilinearly upsampled cosine map. Background is the entire complement
  of the visible object, including green target and agent.
* `localization_summary.json`: accuracy (mean of hits), mean contrast and the
  number of valid frames per backbone, **excluding the initial self-similarity
  frame**. Accuracy is `null` if there are no valid temporal frames; always report
  the valid-frame count alongside it.

For example:

```bash
python spatial_correspondence.py \
  --source-type video --source ../episode_000.mp4 \
  --frames 0 5 10 20 --reference 161 102 \
  --object-mask pusht-color --encoders dino mae --device cuda \
  --output correspondence_results/temporal_with_masks
```

The script warns if the initial reference point is outside the object mask. No
point correspondence or temporal point-error ground truth is inferred from color.

### Manual annotations

PushT's current dataset returns states and a shape name, but no rendered object
segmentation or tracked surface points. The script does not mistake the target,
agent, T-object center, or state coordinates for arbitrary point correspondences.
Supply `--annotations annotations.json` to provide manual ground-truth metrics or
override the automatically derived regions:

```json
{
  "reference_xy": [112, 112],
  "frames": {
    "0": {"mask": "masks/000.png", "correspondence_xy": [112, 112]},
    "5": {"mask": "masks/005.png", "correspondence_xy": [135, 118]},
    "10": {"bbox_xyxy": [100, 90, 180, 170]}
  }
}
```

This is an illustrative schema, not annotations for the included image. Mask paths
are relative to the JSON, nonzero pixels identify the manipulated object, and mask
resolution must exactly match that frame's original RGB. Boxes are `[x0,y0,x1,y1)`
in original coordinates; their metrics are marked `bbox_proxy`. A mask takes
precedence over a box. Missing annotations leave the corresponding CSV fields empty.

* `max_cosine_similarity`: maximum over spatial patch tokens; ties use the first
  patch in row-major order.
* `predicted_x/y`: center of that patch in the original frame.
* `object_localization_hit`: whether the predicted center falls inside the object
  mask or annotated box. Average over annotated nonreference frames for accuracy.
* `object_background_contrast`: mean **bilinearly upsampled** cosine in the object
  region minus mean in its complement. The complement includes agent/target pixels;
  a box is only a region proxy. Empty/full masks yield an empty contrast field.
* `temporal_error_pixels`: Euclidean distance to a supplied corresponding **physical
  point**, including patch quantization. `reference_xy` must match the CLI and the
  first frame must annotate that same point. Annotate rotation-aware point tracks,
  not the object center. The initial error can be nonzero because the query selects
  a cell but the prediction is its center. Errors across different image resolutions
  are in each frame's own pixels; avoid pooling them without a common scale.

## Outputs and interpretation

* `reference_patch.png`: first frame once per backbone, requested point (+) and
  actual cell boundary (green). A transformer's receptive field extends beyond it.
* `comparison.png`: backbone rows, timestamp columns, fixed-reference cosine
  overlays and maxima (black x). Only the first column is within-frame similarity.
* `observations/frame_*.png`: original RGB, shared by all encoders.
* `<encoder>/spatial_similarity.png`: original, reference cell, heatmap and overlay.
* `<encoder>/frame_*_heatmap.png`: individual heatmaps with shared color limits.
* `<encoder>/frame_*_similarity.npy`: float32 `[grid_height,grid_width]` cosine arrays.
* `<encoder>/frame_*_similarity_pixels.npy`: float32 original-resolution maps.
* `<encoder>/reference_feature.npy`: fixed, unnormalized initial patch feature.
* `correspondence_metrics.csv`, `metadata.json`: measurements and provenance.

Color limits default to [-1,1] for every model/frame. `--vmin` and `--vmax` change
display limits globally, never raw arrays; `--alpha` controls overlays. No per-map
min/max scaling is performed. A successful correspondence keeps its peak on the
moving object. Diffuse/background peaks indicate ambiguity; the first column's
self-match near 1 is a sanity check, not temporal performance. **Higher raw cosine
across different architectures is not evidence of better representation quality.**
Use annotated localization/error, contrast, and the spatial distribution jointly.

## SLURM

`spatial_correspondence.sbatch` follows the workspace's `GPU_INFO.md` and batch
examples: `all_usr_prod`, account `cvcs2026`, one 24 GB A5000/RTX6000 GPU, two CPUs,
16 GB host memory, 15 minutes. It avoids reserving a large boost GPU for a small
inference job. Submit from the repository root:

```bash
sbatch spatial_correspondence.sbatch \
  --source-type pusht --source /work/cvcs2026/DINO_WM/pusht_noise/val \
  --trajectory 0 --frames 0 5 10 20 --reference 112 112 \
  --encoders dino clip_encoder siglip mae ijepa \
  --encoder-override mae:local_files_only=true \
  --output correspondence_results/rollout_000
```

`CONDA_ENV_NAME` and `PROJECT_DIR` can override the environment and repository path.
The script preserves SLURM's GPU visibility, uses inference mode and frozen eval
models, releases each encoder before the next, and records peak PyTorch allocated
VRAM. `--cuda-memory-fraction` optionally caps the PyTorch allocator; it is **not** a
hard cap on total process VRAM. SLURM `--mem` controls host RAM, not VRAM. No exact
site VRAM utilization threshold was available in the inspected files; partition
availability and any additional local policy must be checked on the cluster. This
batch script has not been submitted from the local Windows environment.

## Validation

```bash
python -m unittest discover -s tests -p test_spatial_correspondence.py -v
```

Offline tests cover nonsquare geometry and boundaries, raster ordering, fixed
references under motion, negative cosine preservation, optional masks/point error,
model-specific input normalization, frozen eval mode, MAE CLS/projection handling,
unexpected tokens, global-encoder rejection and end-to-end saved outputs with
different frame resolutions. These use small deterministic encoders rather than
claiming pretrained accuracy. Actual model forward passes additionally validate
every selected encoder's grid and dimension at runtime.

Local validation on 2026-10-09: all nine offline tests passed. A CPU smoke run on
the repository's `input_image.png` using cached pretrained weights produced
DINOv2 `[256,384]`, MAE `[196,768]`, and SigLIP `[196,768]`, with reference
self-similarity approximately 1. Results are in `correspondence_results/spatial_smoke/`.
The smoke environment used PyTorch 2.12.0+cpu, Transformers 5.8.0 and Hydra 1.3.2;
these are observations, not new project dependency pins. CLIP/I-JEPA pretrained
passes, actual PushT dataset loading, CUDA and SLURM execution were not tested
locally because the matching weights/package, rollout data or GPU were unavailable.

On 2026-10-10, the automatic-mask extension passed all 13 offline tests, including
color separation, rotation/compression perturbations, missing/ambiguous detections,
manual annotation precedence, localization hits/misses and temporal aggregation.
Masks were also generated for the existing RGB frames 0, 5, 10 and 20; inspection
of the first/last overlays confirmed separation of the gray T from target, agent
and walls. These segmentation-only artifacts are in
`correspondence_results/object_mask_validation/`; this check did not rerun encoders.
