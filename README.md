# Visual backbone comparison for DINO-WM

See [Spatial similarity and temporal correspondence](SPATIAL_CORRESPONDENCE.md)
for the pretrained-feature analysis, preprocessing details, annotations and SLURM usage.

```bash
python spatial_correspondence.py \
    --source-type video --source ../episode_000.mp4 \
    --frames 0 5 10 20 --reference 161 102 \
    --encoders dino mae --device cuda \
    --output correspondence_results/episode_000
```

Select reference coordinates on the manipulated object in the original first frame.
