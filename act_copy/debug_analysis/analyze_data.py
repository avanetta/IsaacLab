"""
Dataset Analysis Script for Diagnosing KL Collapse in CVAE/VAE
==============================================================
Checks for:
  1. Action distribution statistics (mean, std, range per joint)
  2. Action chunk variance — inter-chunk vs intra-chunk
  3. Pairwise chunk similarity heatmap (do all chunks look the same?)
  4. Per-joint temporal profiles (do actions vary across episodes?)
  5. PCA of flattened action chunks (latent geometry / effective dimensionality)
  6. Nearest-neighbour distances (how distinguishable are chunks?)
  7. qpos / action correlation (is state a perfect predictor of actions?)
  8. Chunk overlap / dataset size sanity check
  9. PARAMETER SWEEPS
       9a. Intra/inter-chunk variance  vs chunk_size  (fixed overlap)
       9b. Intra/inter-chunk variance  vs overlap_ratio (fixed chunk_size)
       9c. Effective PCA dimensionality vs chunk_size
       9d. Mean NN distance            vs chunk_size
       9e. # chunks in dataset         vs chunk_size x overlap_ratio (heatmap)
       9f. Per-joint inter-chunk std   vs chunk_size (line per joint)

Usage:
  python analyze_dataset.py --dataset /path/to/merged_data.hdf5 \
                             --num-episodes 58 \
                             --chunk-size 100 \
                             --overlap-ratio 0.6 \
                             --output-dir ./analysis_plots
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_episodes(dataset_path, num_episodes):
    """Return list of dicts with raw numpy arrays per episode."""
    episodes = []
    with h5py.File(dataset_path, "r") as f:
        ids = [f"demo_{i}" for i in range(num_episodes) if f"demo_{i}" in f["data"]]
        for key in tqdm(ids, desc="Loading episodes"):
            demo = f["data"][key]
            ep = {
                "actions": demo["actions"][()],
                "qpos":    demo["states/articulation/robot/joint_position"][()],
            }
            episodes.append(ep)
    return episodes


def build_chunks(episodes, chunk_size, overlap_ratio):
    """Sliding-window chunker; mirrors ChunkedEpisodicDataset logic."""
    stride = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks = []
    for ep in episodes:
        actions = ep["actions"]
        T = actions.shape[0]
        if T < chunk_size:
            pad = np.zeros((chunk_size - T, actions.shape[1]))
            chunks.append(np.concatenate([actions, pad], axis=0))
        else:
            max_start = T - chunk_size
            starts = list(range(0, max_start, stride))
            if not starts or starts[-1] != max_start:
                starts.append(max_start)
            for s in starts:
                chunks.append(actions[s:s + chunk_size])
    return np.stack(chunks, axis=0)  # [N, chunk_size, action_dim]


def chunk_stats(chunks):
    """Return dict of scalar summary statistics for a chunk array."""
    N, T, D = chunks.shape
    flat = chunks.reshape(N, T * D)
    inter_var  = float(flat.var(axis=0).mean())
    intra_var  = float(chunks.var(axis=1).mean())       # variance over time within chunk
    ratio      = inter_var / (intra_var + 1e-9)

    # std of per-chunk mean (one number per joint, then averaged)
    inter_std_per_joint = flat.reshape(N, T, D).mean(axis=1).std(axis=0)

    # PCA effective dim (# PCs for 90% variance)
    n_comp = min(30, N - 1, T * D)
    if n_comp >= 2:
        flat_scaled = StandardScaler().fit_transform(flat)
        pca = PCA(n_components=n_comp)
        pca.fit(flat_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n90 = int(np.searchsorted(cum_var, 0.90)) + 1
    else:
        n90 = 1

    # mean NN distance (subsample for speed)
    n_sample = min(300, N)
    idx = np.random.choice(N, n_sample, replace=False)
    sample = flat[idx]
    dists = cdist(sample, sample, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    mean_nn = float(dists.min(axis=1).mean())

    return {
        "n_chunks":          N,
        "inter_var":         inter_var,
        "intra_var":         intra_var,
        "ratio":             ratio,
        "inter_std_joints":  inter_std_per_joint,   # [D]
        "n90_pca":           n90,
        "mean_nn_dist":      mean_nn,
    }


def savefig(fig, path, title=None):
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Original per-configuration analyses
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_distributions(episodes, out_dir):
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    action_dim  = all_actions.shape[1]

    print(f"\n[1] Action statistics (raw, all timesteps):")
    print(f"    shape  : {all_actions.shape}")
    print(f"    mean   : {all_actions.mean(axis=0).round(4)}")
    print(f"    std    : {all_actions.std(axis=0).round(4)}")
    print(f"    min    : {all_actions.min(axis=0).round(4)}")
    print(f"    max    : {all_actions.max(axis=0).round(4)}")

    std_per_joint = all_actions.std(axis=0)
    dead = np.where(std_per_joint < 0.01)[0]
    if len(dead):
        print(f"  ⚠  Joints with std < 0.01 (near-constant): {dead.tolist()}")
    else:
        print(f"  ✓  All joints have std >= 0.01")

    cols = min(4, action_dim)
    rows = int(np.ceil(action_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    for j in range(action_dim):
        axes[j].hist(all_actions[:, j], bins=60, color="#4C72B0",
                     edgecolor="none", alpha=0.85)
        axes[j].set_title(f"Joint {j}  sigma={std_per_joint[j]:.3f}", fontsize=9)
        axes[j].set_xlabel("action value")
        axes[j].set_ylabel("count")
    for j in range(action_dim, len(axes)):
        axes[j].axis("off")
    savefig(fig, os.path.join(out_dir, "1_action_distributions.png"),
            "Per-joint action distributions (all timesteps)")


def plot_chunk_variance(chunks, out_dir):
    N, T, D = chunks.shape
    flat = chunks.reshape(N, T * D)

    inter_var = flat.var(axis=0).mean()
    intra_var = chunks.var(axis=1).mean()
    ratio = inter_var / (intra_var + 1e-9)

    print(f"\n[2] Chunk variance analysis:")
    print(f"    # chunks         : {N}")
    print(f"    inter-chunk var  : {inter_var:.6f}")
    print(f"    intra-chunk var  : {intra_var:.6f}")
    print(f"    inter/intra ratio: {ratio:.4f}")
    if ratio < 0.05:
        print("  ⚠  All chunks look very similar -> KL collapse expected.")
    else:
        print("  ✓  Inter-chunk variance looks OK.")

    inter_std_per_joint = flat.reshape(N, T, D).mean(axis=1).std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(D), inter_std_per_joint, color="#55A868")
    axes[0].set_title("Std of per-chunk mean action (inter-chunk diversity)")
    axes[0].set_xlabel("Joint index"); axes[0].set_ylabel("std")
    axes[0].axhline(0.01, color="red", linestyle="--", label="threshold 0.01")
    axes[0].legend()

    norms = np.linalg.norm(flat - flat.mean(axis=0, keepdims=True), axis=1)
    axes[1].hist(norms, bins=50, color="#C44E52", edgecolor="none", alpha=0.85)
    axes[1].set_title("Distribution of chunk distances from mean chunk")
    axes[1].set_xlabel("L2 distance"); axes[1].set_ylabel("count")

    savefig(fig, os.path.join(out_dir, "2_chunk_variance.png"),
            "Chunk variance — key KL collapse diagnostic")


def plot_pairwise_similarity(chunks, out_dir, n_sample=200):
    N, T, D = chunks.shape
    idx = np.random.choice(N, min(n_sample, N), replace=False)
    sample = chunks[idx].reshape(len(idx), T * D)

    norms  = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9
    normed = sample / norms
    sim    = normed @ normed.T

    off_diag = sim[np.triu_indices(len(idx), k=1)]
    print(f"\n[3] Pairwise cosine similarity (sample={len(idx)}):")
    print(f"    mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
          f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")
    if off_diag.mean() > 0.95:
        print("  ⚠  Average cosine similarity > 0.95 — chunks nearly identical!")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(f"Pairwise cosine similarity ({len(idx)} random chunks)\n"
                 f"mean off-diag = {off_diag.mean():.3f}")
    ax.set_xlabel("chunk index"); ax.set_ylabel("chunk index")
    savefig(fig, os.path.join(out_dir, "3_pairwise_similarity.png"))


def plot_pca(chunks, out_dir):
    N, T, D = chunks.shape
    flat   = chunks.reshape(N, T * D)
    scaled = StandardScaler().fit_transform(flat)

    n_components = min(50, N - 1, T * D)
    pca = PCA(n_components=n_components)
    pca.fit(scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n99 = int(np.searchsorted(cum_var, 0.99)) + 1

    print(f"\n[4] PCA of action chunks:")
    print(f"    PCs for 90% variance: {n90}")
    print(f"    PCs for 99% variance: {n99}")
    print(f"    Top-5 explained var : {pca.explained_variance_ratio_[:5].round(4)}")
    if n90 <= 2:
        print("  ⚠  Data on <=2-dim manifold -> encoder has nothing to encode.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color="#8172B2", alpha=0.85)
    axes[0].set_title("Explained variance per PC")
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("var ratio")

    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "-o", markersize=3, color="#8172B2")
    axes[1].axhline(0.90, color="orange", linestyle="--", label="90%")
    axes[1].axhline(0.99, color="red",    linestyle="--", label="99%")
    axes[1].set_title(f"Cumulative explained variance  (90%->PC{n90}, 99%->PC{n99})")
    axes[1].set_xlabel("# PCs"); axes[1].set_ylabel("cumulative var ratio")
    axes[1].legend()
    savefig(fig, os.path.join(out_dir, "4a_pca_variance.png"), "PCA of action chunks")

    coords = pca.transform(scaled)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sc = ax2.scatter(coords[:, 0], coords[:, 1], c=np.arange(N),
                     cmap="viridis", alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax2, label="chunk index")
    ax2.set_title("Action chunks — PC1 vs PC2")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "4b_pca_scatter.png")
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓  {path2}")


def plot_temporal_profiles(episodes, out_dir, n_plot=8):
    action_dim = episodes[0]["actions"].shape[1]
    indices = np.random.choice(len(episodes), min(n_plot, len(episodes)), replace=False)

    cols = min(4, action_dim)
    rows = int(np.ceil(action_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    cmap = plt.cm.tab10

    for j in range(action_dim):
        for k, ep_idx in enumerate(indices):
            axes[j].plot(episodes[ep_idx]["actions"][:, j],
                         alpha=0.7, linewidth=0.9,
                         color=cmap(k / max(1, n_plot - 1)),
                         label=f"ep {ep_idx}" if j == 0 else None)
        axes[j].set_title(f"Joint {j}")
        axes[j].set_xlabel("timestep"); axes[j].set_ylabel("action")
    for j in range(action_dim, len(axes)):
        axes[j].axis("off")
    if action_dim > 0:
        axes[0].legend(fontsize=7, ncol=2)
    savefig(fig, os.path.join(out_dir, "5_temporal_profiles.png"),
            f"Per-joint action profiles ({len(indices)} episodes)")


def plot_nn_distances(chunks, out_dir, n_sample=500):
    N, T, D = chunks.shape
    idx = np.random.choice(N, min(n_sample, N), replace=False)
    sample = chunks[idx].reshape(len(idx), T * D)

    dists = cdist(sample, sample, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)

    print(f"\n[5] NN distances (sample={len(idx)}):")
    print(f"    mean={nn_dists.mean():.4f}  std={nn_dists.std():.4f}  "
          f"min={nn_dists.min():.4f}")
    if nn_dists.mean() < 0.5:
        print("  ⚠  Very small NN distances — chunks are near-copies.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nn_dists, bins=50, color="#DD8452", edgecolor="none", alpha=0.85)
    ax.set_title(f"Nearest-neighbour distances\nmean={nn_dists.mean():.3f}")
    ax.set_xlabel("L2 to nearest neighbour"); ax.set_ylabel("count")
    savefig(fig, os.path.join(out_dir, "6_nn_distances.png"))


def plot_qpos_action_correlation(episodes, out_dir):
    from numpy.linalg import lstsq

    qpos_all   = np.concatenate([ep["qpos"]    for ep in episodes], axis=0)
    action_all = np.concatenate([ep["actions"] for ep in episodes], axis=0)

    A = np.hstack([qpos_all, np.ones((qpos_all.shape[0], 1))])
    W, _, _, _ = lstsq(A, action_all, rcond=None)
    pred   = A @ W
    ss_res = ((action_all - pred) ** 2).sum(axis=0)
    ss_tot = ((action_all - action_all.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    print(f"\n[6] Linear R2: qpos -> action per joint: {r2.round(3)}")
    print(f"    mean R2 = {r2.mean():.4f}")
    if r2.mean() > 0.85:
        print("  ⚠  qpos explains >85% of action variance -> z is redundant -> KL collapse.")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(r2)), r2, color="#4878D0", alpha=0.85)
    ax.axhline(0.85, color="red", linestyle="--", label="85% threshold")
    ax.set_title(f"Linear R2: qpos -> action  (mean={r2.mean():.3f})")
    ax.set_xlabel("action joint"); ax.set_ylabel("R2")
    ax.set_ylim(0, 1.05); ax.legend()
    savefig(fig, os.path.join(out_dir, "7_qpos_action_r2.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweeps  (Section 9)
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZES    = [10, 20, 30, 50, 75, 100, 150, 200, 300]
OVERLAP_RATIOS = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9]


def _sweep_chunk_sizes(episodes, overlap_ratio, chunk_sizes, seed=42):
    """Collect stats for each chunk_size at fixed overlap."""
    np.random.seed(seed)
    records = []
    for cs in tqdm(chunk_sizes, desc=f"  chunk_size sweep (overlap={overlap_ratio})"):
        valid = [ep for ep in episodes if ep["actions"].shape[0] >= cs]
        if len(valid) == 0:
            continue
        chunks = build_chunks(valid, cs, overlap_ratio)
        if len(chunks) < 3:
            continue
        s = chunk_stats(chunks)
        s["chunk_size"] = cs
        records.append(s)
    return records


def _sweep_overlap_ratios(episodes, chunk_size, overlap_ratios, seed=42):
    """Collect stats for each overlap_ratio at fixed chunk_size."""
    np.random.seed(seed)
    valid = [ep for ep in episodes if ep["actions"].shape[0] >= chunk_size]
    records = []
    for ov in tqdm(overlap_ratios, desc=f"  overlap sweep (chunk_size={chunk_size})"):
        chunks = build_chunks(valid, chunk_size, ov)
        if len(chunks) < 3:
            continue
        s = chunk_stats(chunks)
        s["overlap_ratio"] = ov
        records.append(s)
    return records


def plot_sweep_chunk_size(episodes, fixed_overlap, out_dir):
    """9a/9c/9d/9f — four panels, all vs chunk_size."""
    print(f"\n[9a-f] Parameter sweep: chunk_size  (overlap={fixed_overlap})")
    records = _sweep_chunk_sizes(episodes, fixed_overlap, CHUNK_SIZES)
    if not records:
        print("  ⚠  No valid records.")
        return

    cs_vals    = [r["chunk_size"]   for r in records]
    inter_vars = [r["inter_var"]    for r in records]
    intra_vars = [r["intra_var"]    for r in records]
    ratios     = [r["ratio"]        for r in records]
    n90s       = [r["n90_pca"]      for r in records]
    nn_dists   = [r["mean_nn_dist"] for r in records]
    n_chunks   = [r["n_chunks"]     for r in records]
    action_dim = episodes[0]["actions"].shape[1]
    joint_stds = np.array([r["inter_std_joints"] for r in records])  # [sweep, D]

    fig = plt.figure(figsize=(16, 14))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 9a  intra/inter variance vs chunk_size
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(cs_vals, inter_vars, "o-", color="#4C72B0", label="inter-chunk var")
    ax.plot(cs_vals, intra_vars, "s-", color="#C44E52", label="intra-chunk var (temporal)")
    ax.set_xlabel("chunk size"); ax.set_ylabel("mean variance")
    ax.set_title("9a  Intra- vs Inter-chunk variance vs chunk_size\n"
                 "(inter << intra -> chunks look alike -> KL collapse)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 9a (right)  ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cs_vals, ratios, "D-", color="#55A868")
    ax2.axhline(0.1, color="red", linestyle="--", label="collapse threshold 0.1")
    ax2.set_xlabel("chunk size"); ax2.set_ylabel("inter / intra variance ratio")
    ax2.set_title("9a  Inter/intra variance ratio vs chunk_size\n"
                  "(ratio < 0.1 -> chunks near-identical)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 9c  effective PCA dim vs chunk_size
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(cs_vals, n90s, "^-", color="#8172B2")
    ax3.set_xlabel("chunk size"); ax3.set_ylabel("# PCs for 90% variance")
    ax3.set_title("9c  Effective PCA dimensionality vs chunk_size\n"
                  "(low -> data on thin manifold -> latent underutilised)")
    ax3.grid(True, alpha=0.3)

    # 9d  mean NN distance vs chunk_size
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(cs_vals, nn_dists, "P-", color="#DD8452")
    ax4.axhline(0.5, color="red", linestyle="--", label="low-diversity threshold 0.5")
    ax4.set_xlabel("chunk size"); ax4.set_ylabel("mean NN distance (L2)")
    ax4.set_title("9d  Mean nearest-neighbour distance vs chunk_size\n"
                  "(small -> chunks near-identical in embedding space)")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # 9e  # chunks vs chunk_size
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(cs_vals, n_chunks, "o-", color="#4878D0")
    ax5.set_xlabel("chunk size"); ax5.set_ylabel("# chunks")
    ax5.set_title(f"9e  Dataset size vs chunk_size  (overlap={fixed_overlap})")
    ax5.grid(True, alpha=0.3)

    # 9f  per-joint inter-chunk std vs chunk_size
    ax6 = fig.add_subplot(gs[2, 1])
    cmap = plt.cm.tab20
    for j in range(action_dim):
        ax6.plot(cs_vals, joint_stds[:, j], "-o", markersize=4,
                 color=cmap(j / max(action_dim - 1, 1)),
                 label=f"J{j}", alpha=0.85)
    ax6.axhline(0.01, color="red", linestyle="--", linewidth=1.2, label="threshold 0.01")
    ax6.set_xlabel("chunk size"); ax6.set_ylabel("inter-chunk std of mean action")
    ax6.set_title("9f  Per-joint inter-chunk diversity vs chunk_size\n"
                  "(joints near 0 carry no signal for the latent)")
    if action_dim <= 12:
        ax6.legend(fontsize=7, ncol=2)
    ax6.grid(True, alpha=0.3)

    savefig(fig, os.path.join(out_dir, "9abcdef_sweep_chunk_size.png"),
            f"Parameter sweep: chunk_size  (overlap_ratio={fixed_overlap})")


def plot_sweep_overlap(episodes, fixed_chunk_size, out_dir):
    """9b — intra/inter variance and # chunks vs overlap_ratio."""
    print(f"\n[9b] Parameter sweep: overlap_ratio  (chunk_size={fixed_chunk_size})")
    records = _sweep_overlap_ratios(episodes, fixed_chunk_size, OVERLAP_RATIOS)
    if not records:
        print("  ⚠  No valid records.")
        return

    ov_vals    = [r["overlap_ratio"] for r in records]
    inter_vars = [r["inter_var"]     for r in records]
    intra_vars = [r["intra_var"]     for r in records]
    ratios     = [r["ratio"]         for r in records]
    n_chunks   = [r["n_chunks"]      for r in records]
    nn_dists   = [r["mean_nn_dist"]  for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(ov_vals, inter_vars, "o-", color="#4C72B0", label="inter-chunk var")
    axes[0].plot(ov_vals, intra_vars, "s-", color="#C44E52", label="intra-chunk var")
    axes[0].set_xlabel("overlap ratio"); axes[0].set_ylabel("mean variance")
    axes[0].set_title("Intra- vs Inter-chunk variance vs overlap_ratio")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    ax_r = axes[0].twinx()
    ax_r.plot(ov_vals, ratios, "D--", color="#55A868", alpha=0.6, label="ratio (right)")
    ax_r.axhline(0.1, color="green", linestyle=":", linewidth=1)
    ax_r.set_ylabel("inter/intra ratio", color="#55A868")
    ax_r.tick_params(axis="y", labelcolor="#55A868")

    axes[1].plot(ov_vals, n_chunks, "o-", color="#4878D0")
    axes[1].set_xlabel("overlap ratio"); axes[1].set_ylabel("# chunks")
    axes[1].set_title(f"Dataset size vs overlap_ratio  (chunk_size={fixed_chunk_size})")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ov_vals, nn_dists, "P-", color="#DD8452")
    axes[2].axhline(0.5, color="red", linestyle="--", label="threshold 0.5")
    axes[2].set_xlabel("overlap ratio"); axes[2].set_ylabel("mean NN distance")
    axes[2].set_title("Mean NN distance vs overlap_ratio\n"
                      "(higher overlap -> more near-duplicate chunks)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    savefig(fig, os.path.join(out_dir, "9b_sweep_overlap_ratio.png"),
            f"Parameter sweep: overlap_ratio  (chunk_size={fixed_chunk_size})")


def plot_sweep_heatmap(episodes, out_dir):
    """
    9e extended — 2D heatmap: rows=chunk_size, cols=overlap_ratio.
    Three heatmaps: # chunks, inter/intra ratio, mean NN distance.
    """
    print(f"\n[9e-heatmap] 2D sweep: chunk_size x overlap_ratio")
    valid_cs = [cs for cs in CHUNK_SIZES
                if any(ep["actions"].shape[0] >= cs for ep in episodes)]

    n_cs = len(valid_cs)
    n_ov = len(OVERLAP_RATIOS)

    grid_nchunks = np.full((n_cs, n_ov), np.nan)
    grid_ratio   = np.full((n_cs, n_ov), np.nan)
    grid_nn      = np.full((n_cs, n_ov), np.nan)

    for i, cs in enumerate(tqdm(valid_cs, desc="  2D heatmap")):
        valid = [ep for ep in episodes if ep["actions"].shape[0] >= cs]
        for j, ov in enumerate(OVERLAP_RATIOS):
            chunks = build_chunks(valid, cs, ov)
            if len(chunks) < 3:
                continue
            s = chunk_stats(chunks)
            grid_nchunks[i, j] = s["n_chunks"]
            grid_ratio[i, j]   = s["ratio"]
            grid_nn[i, j]      = s["mean_nn_dist"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xlab = [f"{o:.2f}" for o in OVERLAP_RATIOS]
    ylab = [str(cs)    for cs in valid_cs]

    def _heatmap(ax, data, title, fmt=".1f", cmap="viridis"):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_ov)); ax.set_xticklabels(xlab, fontsize=8)
        ax.set_yticks(range(n_cs)); ax.set_yticklabels(ylab, fontsize=8)
        ax.set_xlabel("overlap ratio"); ax.set_ylabel("chunk size")
        ax.set_title(title)
        valid_vals = data[~np.isnan(data)]
        mean_val = valid_vals.mean() if len(valid_vals) else 0
        for ii in range(n_cs):
            for jj in range(n_ov):
                v = data[ii, jj]
                if not np.isnan(v):
                    ax.text(jj, ii, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v < mean_val else "black")

    _heatmap(axes[0], grid_nchunks, "# chunks in dataset",       fmt=".0f", cmap="Blues")
    _heatmap(axes[1], grid_ratio,   "inter/intra variance ratio", fmt=".3f", cmap="RdYlGn")
    _heatmap(axes[2], grid_nn,      "mean NN distance",           fmt=".2f", cmap="RdYlGn")

    savefig(fig, os.path.join(out_dir, "9e_heatmap_chunk_overlap.png"),
            "2D sweep: chunk_size x overlap_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_report(episodes, chunks):
    N, T, D = chunks.shape
    flat = chunks.reshape(N, T * D)
    inter_var = flat.var(axis=0).mean()
    intra_var = chunks.var(axis=1).mean()
    ratio = inter_var / (intra_var + 1e-9)

    all_actions   = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    std_per_joint = all_actions.std(axis=0)
    dead_joints   = (std_per_joint < 0.01).sum()

    print("\n" + "=" * 65)
    print("  SUMMARY — KL Collapse Root Cause Checklist")
    print("=" * 65)

    checks = [
        (dead_joints > 0,
         f"{dead_joints} joints have near-zero std (<0.01) — dead dimensions",
         "Remove or ignore those joints; they pollute the loss without adding signal."),
        (ratio < 0.1,
         f"Inter/intra chunk variance ratio = {ratio:.4f} (< 0.1) — low diversity",
         "Increase episode variety, reduce overlap_ratio, or shrink chunk_size."),
        (N < 500,
         f"Only {N} chunks — dataset too small for a meaningful latent space",
         "Collect more data or increase overlap to inflate dataset size."),
    ]

    any_issue = False
    for bad, bad_msg, fix in checks:
        if bad:
            any_issue = True
            print(f"\n  X  {bad_msg}")
            print(f"      -> {fix}")

    if not any_issue:
        print("\n  OK  No obvious data-side red flags found.")
        print("      KL collapse is likely a model/training issue.")
        print("      Suggestions:")
        print("        * Use free-bits  (lambda ~= 0.5-2.0 nats per dim)")
        print("        * Warm up beta slowly over >=50 epochs from 0 -> 1")
        print("        * Initialise fc_logvar bias to ~-2")
        print("        * Make encoder deeper / decoder shallower")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str,   required=True)
    parser.add_argument("--num-episodes",  type=int,   default=58)
    parser.add_argument("--chunk-size",    type=int,   default=100)
    parser.add_argument("--overlap-ratio", type=float, default=0.6)
    parser.add_argument("--output-dir",    type=str,   default="./analysis_plots")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--skip-sweeps",   action="store_true",
                        help="Skip parameter sweeps (faster for quick checks)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nDataset  : {args.dataset}")
    print(f"Episodes : {args.num_episodes}")
    print(f"Out dir  : {args.output_dir}\n")

    # Load
    episodes = load_all_episodes(args.dataset, args.num_episodes)
    print(f"Loaded {len(episodes)} episodes.")

    # Build chunks at requested configuration
    chunks = build_chunks(episodes, args.chunk_size, args.overlap_ratio)
    print(f"Built {len(chunks)} chunks of shape {chunks.shape[1:]}")

    # Per-configuration analyses
    plot_action_distributions(episodes, args.output_dir)
    plot_chunk_variance(chunks, args.output_dir)
    plot_pairwise_similarity(chunks, args.output_dir)
    plot_pca(chunks, args.output_dir)
    plot_temporal_profiles(episodes, args.output_dir)
    plot_nn_distances(chunks, args.output_dir)
    plot_qpos_action_correlation(episodes, args.output_dir)

    # Parameter sweeps
    if not args.skip_sweeps:
        print("\n" + "-" * 55)
        print("  Running parameter sweeps (add --skip-sweeps to skip)")
        print("-" * 55)
        plot_sweep_chunk_size(episodes, fixed_overlap=args.overlap_ratio,  out_dir=args.output_dir)
        plot_sweep_overlap(episodes,    fixed_chunk_size=args.chunk_size,  out_dir=args.output_dir)
        plot_sweep_heatmap(episodes,    out_dir=args.output_dir)
    else:
        print("\n  (Parameter sweeps skipped)")

    print_summary_report(episodes, chunks)


if __name__ == "__main__":
    main()