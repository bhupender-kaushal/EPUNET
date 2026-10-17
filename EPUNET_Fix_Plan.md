# EPUNET — LIDC Fix Plan (implementation spec for code mode)

Repo: `bhupender-kaushal/EPUNET` @ `e75c3b0`. Scope: the LIDC pipeline only:
`load_LIDC_data.py` → `hierarchical2.py` → `train2.py` → `test2.py`.
SNEMI3D and Cityscapes share bug F1 and are ported **after** LIDC is verified (see Step D).

Rules for code mode:
- Make **incremental edits only**. Never rewrite whole files.
- Give each change a short *what / why*.
- Keep the old behaviour reachable via a flag only where noted.

---

## Execution order

| Step | What | Retrain? | Purpose |
|---|---|---|---|
| **A** | Evaluation fixes E1–E5 in `test2.py` | No | Re-score the **existing** checkpoint to get the honest current numbers |
| **B** | Model / training fixes F1–F5 | Yes | Make the hierarchical VAE + EDL objective correct |
| **C** | Clean runs: one config, from scratch, seeds {0, 1, 2} | Yes | Reproducible mean ± std for the paper |
| **D** | Port F1 to `SNIMED/hierarchical_model_snemi3d.py` and `Cityscapes/hierarchical_model_cityscapes_new.py` | Later | Same KL bug exists there |

Do Step A **before** touching `hierarchical2.py`, so that the old checkpoint is evaluated with the model code it was trained with.

---

## STEP A — Evaluation fixes (`test2.py`)

### E1. Hungarian-matched IoU: duplicate both sets to their LCM (Kohl et al. 2019 protocol)
**Where:** `hungarian_matched_iou` (≈ lines 60–75). Replace the body after the `M == 0 or N == 0` guard:
```python
from math import lcm
L = lcm(M, N)                                     # 16 for 4 GTs x 16 samples
gts_rep     = [gts[i % M] for i in range(L)]
samples_rep = [samples[j % N] for j in range(L)]
cost = -np.array([[kohl_iou(s, g) for s in samples_rep] for g in gts_rep])
r, c = linear_sum_assignment(cost)
return float(-cost[r, c].mean())
```
**Why:** on a rectangular 4×16 matrix, the current code matches each grader to its best sample and discards 12 samples. That is best-of-N, which inflates the score (0.80 vs the paper's 0.53). The LCM duplication forces every sample to be matched.

### E2. IoU_rec: exclude correctly predicted absences (NaN-mean)
**Where:** add a helper next to `kohl_iou`, and use it **only** for IoU_rec (≈ line 142). Hungarian IoU keeps `kohl_iou`.
```python
def iou_rec_pair(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return np.nan                     # correctly predicted absence -> excluded
    if p.sum() == 0 or g.sum() == 0:
        return 0.0                        # wrongly predicted absence -> penalised
    return np.logical_and(p, g).sum() / np.logical_or(p, g).sum()
```
- Per slice: `avg_slice_iou_rec = np.nanmean(slice_iou_rec_list)`. If all four values are NaN, store NaN.
- Final summary: aggregate with `np.nanmean`.

**Why:** the paper defines IoU_rec with a NaN-mean over correct absences. Counting empty–empty as 1.0 inflates IoU_rec on LIDC, where most grader masks are empty.

### E3. Threshold: no hand-tuned 0.505
**Where:** `evaluate_full_dataset(..., threshold=0.505)` and the other two analysis functions. Set a module constant `THRESHOLD = 0.5` and use it everywhere; also in `train2.py` validation (line ≈190).
**Why:** a threshold tuned on the test set is leakage. If a different threshold is wanted, select it on the **validation** set and report it.

### E4. Evaluate the checkpoint that was actually selected
**Where:** `load_best_checkpoint` (≈ line 407) currently loads `model_latest.pth`. Add a parameter `which="best"` that maps to `model_best.pth` / `model_latest.pth`. Print which file was loaded and the epoch stored in it.
**Why:** the reported numbers currently come from the last epoch, while the code and logs say "best". The paper must state which one is used.

### E5. Uncertainty sanity check (AUROC + ECE + decomposition)
**Where:** inside the slice loop of `evaluate_full_dataset`, for slices with `num_tumor_gts > 0`, collect flattened arrays. Compute the metrics after the loop.
```python
# per slice (lesion slices only)
consensus = (np.sum(gt_list, axis=0) >= 2).astype(np.uint8)
grader_frac = np.mean(gt_list, axis=0)                         # 0, .25, .5, .75, 1
error = (mean_recon.astype(np.uint8) != consensus).astype(np.uint8)
aleatoric = np.std(np.stack(sample_probs), axis=0)             # spread across latent samples
unc_store['u'].append(expected_epistemic.ravel())
unc_store['alea'].append(aleatoric.ravel())
unc_store['err'].append(error.ravel())
unc_store['p'].append(mean_prob_map.ravel())
unc_store['y'].append(consensus.ravel())
unc_store['disagree'].append((grader_frac * (1 - grader_frac)).ravel())  # 0 when graders agree
```
After the loop (`from sklearn.metrics import roc_auc_score`), compute and write to `global_summary.txt`:
- `AUROC(u, err)`: does epistemic uncertainty flag errors? **Key number.**
- `AUROC(alea, err)`: the same for the latent-sample spread.
- Mean `u` on correct vs wrong pixels.
- `ECE(p, y)` with 15 equal-width bins.
- `Spearman(u, disagree)` and `Spearman(alea, disagree)`. Expectation: aleatoric should correlate with grader disagreement, epistemic much less. If `u` tracks disagreement as strongly as `alea` does, the EDL head is absorbing aleatoric ambiguity (bug F4).

**Why:** this decides whether the epistemic half of the paper holds. A mean u ≈ 0.7 with no separation would mean the evidential head is not informative.

### E6. Loading the old checkpoint after Step B (helper)
Bug F2 removes 4 unused `dist_convs` layers, so old checkpoints will contain extra keys. Add to `load_best_checkpoint`:
```python
sd = {k: v for k, v in sd.items() if k in net.state_dict()}   # drop dead dist_convs.4-7
net.load_state_dict(sd, strict=True)
```
**Why:** this lets you keep evaluating the old run after the fix. The dropped layers were never used in `forward`, so the outputs are identical.

**Step A deliverable:** `global_summary.txt` for the existing checkpoint, with GED, IoU_rec (E2), Hungarian IoU (E1), the AUROC/ECE block, and the selected checkpoint stated.

---

## STEP B — Model and training fixes

### F1. 🔴 Hierarchical KL: condition the prior on the posterior's latent path (`hierarchical2.py`, `forward_train_step`)
**Problem:**
- `prior_out = self.prior(image)` builds prior levels 2–4 from the prior's own samples, while the posterior builds them from its own samples. The KL then compares two unrelated chains.
- The MC loop draws `dist.rsample()` per level from distributions that were conditioned on a *different* earlier sample, so z₁…z₄ are mutually inconsistent.
- The deterministic pass has the same issue: `dist.mean` per level is taken from the sampled chain, not from a chain of means.

**Change** (only these parts of `forward_train_step`):
```python
posterior_in  = torch.cat([mask_one_hot, image], dim=1)
posterior_out = self.posterior(posterior_in, use_mean=False)
z_post        = posterior_out['used_latents']                      # one consistent top-down sample
prior_cond    = self.prior(image, z_q=z_post)                      # p(z_i | z_<i (posterior), x)
vae_kl_loss   = self.calculate_hierarchical_kl(prior_cond['distributions'],
                                               posterior_out['distributions'])

# MC loop
for k in range(n_samples):
    if k == 0:
        prior_z_out = prior_cond                                   # reuse: already decoded with z_post
    else:
        z_k = self.posterior(posterior_in, use_mean=False)['used_latents']
        prior_z_out = self.prior(image, z_q=z_k)
    logits_stoch = self.f_comb(prior_z_out['encoder_features'], prior_z_out['decoder_features'])
    ...  # unchanged loss code

# Deterministic pass: a chain of posterior MEANS
z_mean = self.posterior(posterior_in, use_mean=True)['used_latents']
prior_z_mean_out = self.prior(image, z_q=z_mean)
```
- Remove the now-unused initial `prior_out = self.prior(image, use_mean=False)`.

**Why:** in ladder / hierarchical VAEs (Sønderby et al., NeurIPS 2016; Kohl et al. 2019; Vahdat & Kautz, NVAE, NeurIPS 2020), each prior level is evaluated along the posterior's sampled path. Only then is the summed KL the correct ELBO term, and only then does the prior learn the posterior's choices. This change directly affects sample diversity, and therefore GED.

**Cost:** one extra posterior pass per extra MC sample. A later optimisation is to cache encoder features, as the Cityscapes core already allows.

**Check after the change:** `vae_kl_loss >= 0` on every step. Over the first epochs, KL per level should be non-zero for more than one level.

### F2. Duplicate `dist_convs` + missing zero-init (`hierarchical2.py`, `HierarchicalCore.__init__`)
**Problem:** the first loop (≈ lines 124–130) appends 4 convs. A second loop (≈ lines 132–139) appends 4 more, zero-initialised, with the wrong input channels. `forward` only uses indices 0–3, so the zero-init never applies and 4 dead layers receive L2.

**Change:** delete the second loop. Inside the first loop, zero-init the layer:
```python
dist_layer = nn.Conv2d(curr_dec_channels, 2 * latent_dim, kernel_size=1)
nn.init.zeros_(dist_layer.weight); nn.init.zeros_(dist_layer.bias)
self.dist_convs.append(dist_layer)
```
**Why:** the intended initialisation (prior = posterior = N(0, 1) at the start) actually takes effect, and the dead parameters disappear. Old checkpoints load via E6.

### F3. Anneal the EDL KL term (`hierarchical2.py` + `train2.py`)
**Change:**
- Add an argument `edl_kl_weight=1.0` to `forward_train_step`, and use `edl_kl_weight * masked_kl_edl` in the final loss.
- In `train2.py`, per epoch: `edl_kl_weight = min(1.0, epoch / 10.0)`.

**Why:** Sensoy et al. (NeurIPS 2018) anneal the Dirichlet KL as λₜ = min(1, t/10). Applying it at full weight from step 0 pushes the evidence towards collapse before the classifier has learned anything. This is the minimal faithful baseline; the multi-grader-aware variant is Phase 2 research, not part of this fix.

### F4. ⚠️ Not a fix yet — flag only
The EDL KL and AU losses use the one-hot label of **one random grader**. At pixels where graders disagree, this penalises evidence for the other grader's answer. **Do not change this in Step B.** It is the Phase 2 research question, and E5 measures its effect. Add a `# TODO(phase2)` comment at the `unreduced_evidential_loss` call in the deterministic pass.

### F5. Reproducible training (`train2.py`)
**Changes:**
1. `RESUME_DIR = None` by default. When resuming, load **`model_latest.pth`**, not `model_best.pth`, and also restore `best_val_ged` and `lambda_geco`.
2. Read `seed` from `argparse` (`--seed`, default 0).
3. At startup, write every hyperparameter to `config.json` in the checkpoint directory: kappa, lambda_lr, lambda_min, lambda_max, lr, batch size, epochs, latent_dims, channels, n_samples, OHEM k, THRESHOLD, EDL anneal epochs. Store the same dict in each checkpoint.
4. Fix the stale `kappa` comment (the code says 0.27, the comment says 0.22). Keep **one** value for the whole run.
5. Validation threshold: use `THRESHOLD = 0.5` (E3).
6. Per-epoch log line: also log mean train VAE-KL, EDL-KL, AU, and `edl_kl_weight`.

**Why:** the paper must report a model trained from scratch with one stated config, over several seeds (HPU-Net used 10). Resuming from "best" with changed hyperparameters cannot be reproduced.

### F6. Guard against silently missing GT files (`load_LIDC_data.py`, `LIDC_IDRI_Folder`)
**Change:** in `__init__`, count the images for which any `gt/..._l{0..3}.png` is missing. Print the count, and `raise` if it is above 0, unless `allow_missing_gt=True` is passed. Keep the `zeros_like` fallback only behind that flag.

**Why:** a wrong path currently turns every label into an empty mask with no warning.

### Not changed (deliberately)
- **GED implementation:** correct, standard estimator, same d(empty, empty) = 0 convention as Kohl et al.
- **Data layout, 180→128 crop, patient-split folders:** match the released LIDC crops. Confirm the provenance of `data/` in the paper.
- **Evidential loss formulas** (expected CE, Dirichlet KL, AU): correct.
- **Loss composition** (GECO + anchor + hinge + OHEM): left as is for now. Each term gets an ablation in Phase 2, not a silent change here.

---

## STEP C — Clean runs
- Train from scratch with the post-Step-B code: seeds 0, 1, 2, one `config.json`, no mid-run edits.
- Evaluate `model_best.pth` (selected on validation GED) with the Step A evaluation.
- Report mean ± std for GED, IoU_rec, Hungarian IoU (full + 4-grader subset), and AUROC(u, err).
- The comparison of Step A numbers (old model) vs Step C numbers (fixed model) is the first ablation.

## STEP D — Port to the other datasets (after C)
Apply F1 to `SNIMED/hierarchical_model_snemi3d.py` (≈ lines 335–349) and `Cityscapes/hierarchical_model_cityscapes_new.py` (≈ lines 334–347). The pattern is identical: `prior_out = self.prior(image)` is used for the KL, and per-level `rsample()` is used in the MC loop. The Cityscapes-specific issues (aligned GED, missing `label_mode`, the broken static loader) are handled in a separate plan.

---

## Smoke tests for code mode (run after each step)
1. **Step A:** run `test2.py` on the first 20 test slices. Expect Hungarian IoU < the old value, IoU_rec ≤ the old value, and the AUROC/ECE block printed.
2. **Step B:** one training batch on CPU or GPU. Check that the loss is finite, `vae_kl_loss >= 0`, and `len(net.prior.dist_convs) == 4`. The old checkpoint should load via E6, and `net.sample()` output should be unchanged versus before F2.
3. **Step B:** 1 epoch with `epochs=1`. Check that `config.json` is written and that the log contains `edl_kl_weight=0.0`.

References: Kohl et al. 2019 (HPU-Net, arXiv:1905.13077); Sønderby et al., NeurIPS 2016 (Ladder VAE); Vahdat & Kautz, NeurIPS 2020 (NVAE); Sensoy et al., NeurIPS 2018 (EDL); Rezende & Viola 2018 (GECO).
