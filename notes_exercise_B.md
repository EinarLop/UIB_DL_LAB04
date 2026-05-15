# Exercise B — LR Scheduler Notes

Reference for the written analysis (cell-16). Covers the rationale for each scheduler choice.

---

## Run 2 — CosineAnnealingLR

### `T_max = 30`

`T_max` is the number of `scheduler.step()` calls it takes for the LR to complete a half-cycle (cosine going from starting `lr` down to `eta_min`).

In our `train()` function, `scheduler.step()` is called **once per epoch** (after the inner batch loop). Since training runs for 30 epochs, we set `T_max = 30` so the LR reaches its minimum exactly on the last epoch.

> ⚠️ Common mistake: setting `T_max = epochs` when stepping per batch (or vice versa). The LR then either restarts mid-training or never reaches the floor. Always match the unit to where `scheduler.step()` is called.

### `eta_min = 3e-6`

`eta_min` is the floor the cosine decays toward. Two principles:

1. **Much smaller than starting `lr`** (which is `3e-4`). `3e-6` is 100× smaller — gives the cosine a meaningful range to anneal over.
2. **Non-zero**. Exactly zero kills learning at the end. A tiny positive floor (~1–2 orders of magnitude below `lr`) is standard.

### Expected LR trajectory

| Epoch | LR        |
|-------|-----------|
| 0     | 3.0e-4    |
| 15    | ~1.5e-4   |
| 29    | 3e-6      |

---

## Run 3 — OneCycleLR

### Why this scheduler fits diffusion training

Diffusion training has a unique characteristic: **every batch sees a random timestep `t`**, which means the noise level (and thus the prediction difficulty) varies wildly across batches early on. The loss signal is noisy.

OneCycleLR addresses this with two phases:

1. **Warm-up phase** — LR ramps *up* from a small value to `max_lr` over the first ~30% of training. Stabilizes early epochs when:
    - Model weights are random
    - Gradient magnitudes are large and noisy
    - A large LR right at the start could push the model into a bad region
2. **Annealing phase** — cosine decay from `max_lr` down to a very small final LR. Same fine-tuning benefit as CosineAnnealingLR.

### Justification (story for the written analysis)

> "I chose OneCycleLR because diffusion training has high gradient variance in early epochs (random `t` per batch). Warm-up stabilizes this initial phase, and the subsequent cosine decay matches the benefits of Run 2. This lets me isolate whether warm-up specifically helps over pure cosine annealing."

Run 2 vs Run 3 is a real comparison: **does warm-up matter on top of cosine decay?**

### Configuration

```python
onecycle_scheduler = lambda opt: optim.lr_scheduler.OneCycleLR(
    opt, max_lr=3e-4, total_steps=30, pct_start=0.3
)
```

- `max_lr=3e-4` — peak LR, same as the baseline starting LR for fair comparison.
- `total_steps=30` — matches the number of `scheduler.step()` calls (per-epoch stepping, Option B).
- `pct_start=0.3` — the first 30% of steps (~9 epochs) are warm-up; the remaining 70% is cosine decay.

### ⚠️ Important note on stepping granularity

`OneCycleLR` is designed to be stepped **per batch**, not per epoch.

- **Option A (cleaner):** set `total_steps` to total batches across all epochs and call `scheduler.step()` inside the batch loop.
- **Option B (used here):** keep stepping per epoch with `total_steps=epochs`. Less granular but works fine for 30 epochs.

We use Option B for simplicity.

---

## Alternatives considered (and why not)

- **CosineAnnealingWarmRestarts** — periodic restarts can help escape shallow local minima. Good story, but less clearly motivated for only 30 epochs.
- **ReduceLROnPlateau** — needs a stable validation metric. With 30 noisy epochs on training loss it won't trigger meaningfully.
- **StepLR / ExponentialLR** — too blunt; the smooth annealing of cosine generally outperforms discrete drops or pure exponential decay for diffusion models.
