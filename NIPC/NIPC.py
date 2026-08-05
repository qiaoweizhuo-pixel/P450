#!/usr/bin/env python3
"""
N-terminal Importance Profile Clustering (v1.4.1)

Clusters P450 proteins based on the position of their N-terminal
importance peaks, using per-protein importance CSVs from either
PIVOT/SVM (*_full_importance.csv) or DIVA (*_residue_importance.csv).

Changes in v1.4.1:
  - load_importance_files: fixed silent deduplication when a gene
    exists in both *_full_importance.csv and *_residue_importance.csv
    formats.  The first occurrence is now kept and skipped duplicates
    are reported explicitly.

Changes in v1.4:
  - Added --mode {diva,pivot} to switch between two peak criteria:
      diva:  max(N-term) >= peak_ratio * max(whole)
             (asks: "does the N-terminus have a signal close to the
              global maximum?" — suited for DIVA's smooth, decaying
              per-protein-normalised profiles)
      pivot: max(N-term) >= peak_factor * mean(N-term)
             (asks: "does the N-terminus have a sharp isolated peak
              above its own baseline?" — suited for PIVOT's sparse
              high-importance-residue profiles)
  - is_nterm_driven, run_clustering, and the CLI parser all accept
    the new parameters while remaining backward-compatible.

Changes in v1.3:
  - load_importance_files now accepts both PIVOT/SVM output
    (*_full_importance.csv, column: "importance") and DIVA output
    (*_residue_importance.csv, column: "final_score_norm").
  - Auto-detects the importance column and maps it uniformly.

Changes in v1.2:
  - resample_profile gained direction enforcement (flip profiles with
    centre of mass in the second half).  **This option must remain
    disabled (enforce_direction=False) when analysing DIVA or PIVOT
    N-terminal profiles**, because N→C direction carries biological
    meaning for both metrics.
  - label_clusters checks within-cluster peak dispersion and warns
    when std > 0.30, using median peak instead of mean profile peak.
  - Default auto-cluster minimum raised from 2 to 3.

Workflow:
  1. Read all importance CSV files from input folder.
  2. Determine the N‑terminal window for each protein.
  3. Separate proteins with clear N‑terminal peaks from those with
     diffuse / weak N‑terminal importance ("Weak N‑terminal dependency").
  4. Hierarchically cluster the N‑terminal‑driven proteins based on
     their resampled importance profiles.
  5. Label each cluster according to where the mean peak falls within
     the N‑terminal window (e.g. "Early‑N", "Mid‑N").
  6. Export cluster assignments, per‑cluster gene lists, profile plots,
     a summary heatmap, and CSV data tables for all figures.

Usage:
  # DIVA profiles (default)
  python NIPC.py \
      --importance-dir ./diva_results \
      --output-dir ./diva_clusters \
      --mode diva --peak-ratio 0.85

  # PIVOT profiles
  python NIPC.py \
      --importance-dir ./pivot_results \
      --output-dir ./pivot_clusters \
      --mode pivot --peak-factor 1.5
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ======================== Helper Functions ================================

# Column names recognised as importance values, in order of preference.
_IMPORTANCE_CANDIDATES = ["importance", "final_score_norm", "final_score"]


def load_importance_files(input_dir: str) -> dict:
    """
    Read importance CSV files from a directory.

    Supports two formats:
      - PIVOT/SVM output:  *_full_importance.csv    (column: importance)
      - DIVA output:       *_residue_importance.csv  (column: final_score_norm)

    The importance column is auto-detected and standardised to "importance".

    Returns
    -------
    dict : {gene_name: pd.DataFrame}
        Each DataFrame has columns: position, amino_acid, importance
    """
    import_dir = Path(input_dir)

    # Match both naming conventions
    files = (sorted(import_dir.glob("*_full_importance.csv")) +
             sorted(import_dir.glob("*_residue_importance.csv")))

    if not files:
        raise FileNotFoundError(
            f"No *_full_importance.csv or *_residue_importance.csv files "
            f"found in {input_dir}")

    data = {}
    duplicates = []                              # v1.4.1 FIX
    imp_col = None
    for fpath in files:
        # Strip either suffix to get the gene name
        gene = fpath.stem
        for suffix in ["_full_importance", "_residue_importance"]:
            if gene.endswith(suffix):
                gene = gene[:-len(suffix)]
                break

        # -- v1.4.1 FIX: skip duplicate gene names -----------------------------
        if gene in data:
            duplicates.append((gene, fpath.name))
            continue
        # ---------------------------------------------------------------------- 

        df = pd.read_csv(fpath)

        # Auto-detect importance column
        imp_col = None
        for candidate in _IMPORTANCE_CANDIDATES:
            if candidate in df.columns:
                imp_col = candidate
                break
        if imp_col is None:
            raise ValueError(
                f"File {fpath.name} must contain one of: "
                f"{', '.join(_IMPORTANCE_CANDIDATES)}")
        if "position" not in df.columns:
            raise ValueError(
                f"File {fpath.name} must contain a 'position' column")

        # Standardise: map detected column → "importance"
        if imp_col != "importance":
            df["importance"] = df[imp_col]

        data[gene] = df

    print(f"Loaded {len(data)} importance profiles "
          f"(importance column: '{imp_col}').")
    if duplicates:                               # v1.4.1 FIX
        print(f"  Note: {len(duplicates)} gene(s) found in both formats; "
              f"kept first occurrence:")
        for g, fn in duplicates:
            print(f"    {g}  (skipped {fn})")
    return data


def compute_nterm_length(protein_length: int, proportion: float = 0.15,
                         min_len: int = 30, max_len: int = 100) -> int:
    """Determine the N-terminal window size for a protein."""
    raw = int(np.ceil(protein_length * proportion))
    return max(min_len, min(raw, max_len))


def extract_nterm_profile(df: pd.DataFrame, nterm_len: int) -> np.ndarray:
    """Extract raw importance values for the first *nterm_len* positions."""
    subset = df[df["position"] <= nterm_len]
    subset = subset.sort_values("position")
    profile = subset["importance"].values.astype(float)
    if len(profile) < nterm_len:
        profile = np.pad(profile, (0, nterm_len - len(profile)),
                         mode='edge')
    return profile


# ======================================================================
#  CLASSIFICATION  (changed in v1.4 — mode‑dependent peak criterion)
# ======================================================================

def is_nterm_driven(df: pd.DataFrame, nterm_len: int,
                    weak_threshold: float = 1.0,
                    peak_ratio: float = 0.85,
                    peak_factor: float = 1.5,
                    mode: str = "diva") -> bool:
    """
    Determine whether a protein has a clear N-terminal importance signal.

    Both criteria must be met:

    1.  Concentration criterion (same for both modes)
        mean(N‑term) / mean(whole) ≥ weak_threshold

    2.  Peak criterion (mode‑dependent)
        mode="diva":  max(N‑term) ≥ peak_ratio × max(whole)
        mode="pivot": max(N‑term) ≥ peak_factor × mean(N‑term)

    The per‑mode rationale:

    DIVA profiles are smooth and per‑protein min‑max normalised; asking
    whether the N‑terminus captures a large fraction of the global maximum
    (peak_ratio, default 0.85) tests whether the N‑terminus is the dominant
    signal carrier without requiring it to be the *single* most important
    residue (a position‑encoding artifact could nudge the global max one or
    two residues outside the window).

    PIVOT profiles contain sharp, sparse peaks (dropout‑based attribution);
    asking whether the N‑terminal maximum exceeds the N‑terminal mean by a
    certain factor (peak_factor, default 1.5) tests whether the peak is
    genuinely discriminative rather than just riding a high baseline.

    Parameters
    ----------
    df : pd.DataFrame
    nterm_len : int
    weak_threshold : float
        Concentration ratio threshold.
    peak_ratio : float
        [diva mode] Fraction of global max the N‑term max must reach.
    peak_factor : float
        [pivot mode] Factor by which N‑term max must exceed N‑term mean.
    mode : str
        "diva" or "pivot".

    Returns
    -------
    bool
    """
    all_imp = df["importance"].values.astype(float)
    nterm_imp = all_imp[:min(nterm_len, len(all_imp))]

    # Criterion 1 — concentration
    concentration = (np.mean(nterm_imp)
                     / (np.mean(all_imp) + 1e-10))

    # Criterion 2 — mode‑dependent
    if mode == "diva":
        global_max = np.max(all_imp)
        nterm_max = np.max(nterm_imp)
        has_peak = (nterm_max >= peak_ratio * global_max)
    elif mode == "pivot":
        has_peak = (np.max(nterm_imp)
                    >= peak_factor * np.mean(nterm_imp))
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'diva' or 'pivot'")

    return (concentration >= weak_threshold) and has_peak


def resample_profile(profile: np.ndarray, target_len: int = 50,
                     smooth_sigma: float = 1.0,
                     enforce_direction: bool = False):
    """
    Resample a variable-length importance profile to a fixed length,
    with optional Gaussian smoothing, and normalise to [0, 1].


    Parameters
    ----------
    profile : np.ndarray, shape (L,)
    target_len : int
    smooth_sigma : float
    enforce_direction : bool
        If True, profiles whose centre of mass falls in the second
        half are reversed.  **Keep False for DIVA / PIVOT analyses**
        because N→C direction carries biological meaning.

    Returns
    -------
    resampled : np.ndarray, shape (target_len,)
    flipped : bool
        Whether the profile was flipped.
    """
    L = len(profile)
    if L < 4:
        interp = np.interp(np.linspace(0, 1, target_len),
                           np.linspace(0, 1, L), profile)
        rmin, rmax = interp.min(), interp.max()
        if rmax - rmin > 1e-10:
            interp = (interp - rmin) / (rmax - rmin)
        else:
            interp = np.zeros(target_len)
        flipped = False
        if enforce_direction:
            weights = np.maximum(interp, 0)
            if weights.sum() > 1e-10:
                centre = np.average(np.arange(target_len),
                                    weights=weights)
                if centre > target_len / 2:
                    interp = interp[::-1]
                    flipped = True
        return interp, flipped

    smoothed = gaussian_filter1d(profile.astype(float), sigma=smooth_sigma)
    x_old = np.linspace(0, 1, L)
    x_new = np.linspace(0, 1, target_len)
    interp_obj = interp1d(x_old, smoothed, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
    resampled = interp_obj(x_new)

    rmin, rmax = resampled.min(), resampled.max()
    if rmax - rmin > 1e-10:
        resampled = (resampled - rmin) / (rmax - rmin)
    else:
        return np.zeros(target_len), False

    flipped = False
    if enforce_direction:
        weights = np.maximum(resampled, 0)
        if weights.sum() > 1e-10:
            centre = np.average(np.arange(target_len), weights=weights)
            if centre > target_len / 2:
                resampled = resampled[::-1]
                flipped = True

    return resampled, flipped


def find_peak_position(profile: np.ndarray) -> float:
    """Return the relative position (0–1) of the maximum in *profile*."""
    idx = np.argmax(profile)
    return idx / max(len(profile) - 1, 1)


def label_clusters(profiles: np.ndarray, cluster_labels: np.ndarray
                   ) -> dict:
    """
    Assign human-readable names to clusters based on the mean profile
    peak position, with a check for within-cluster bimodality.

    When a cluster shows high peak dispersion (std > 0.30), the
    median individual peak is used instead of the mean profile peak,
    and a warning is printed.

    Returns
    -------
    dict : {cluster_id: label_string}
    """
    unique = sorted(set(cluster_labels))
    mean_peaks = {}
    warnings_list = []

    for c in unique:
        mask = cluster_labels == c
        cl_profs = profiles[mask]
        mean_prof = cl_profs.mean(axis=0)

        individual_peaks = [find_peak_position(p) for p in cl_profs]
        peak_std = np.std(individual_peaks)

        if peak_std > 0.30:
            warnings_list.append(
                f"  Warning: Cluster {c} has high peak dispersion "
                f"(std={peak_std:.2f}, n={cl_profs.shape[0]}). "
                f"Consider increasing --n-clusters. "
                f"Using median peak for labelling.")
            mean_peaks[c] = np.median(individual_peaks)
        else:
            mean_peaks[c] = find_peak_position(mean_prof)

    if warnings_list:
        for w in warnings_list:
            print(w)

    sorted_clusters = sorted(unique, key=lambda c: mean_peaks[c])
    n = len(sorted_clusters)

    label_map = {}
    if n == 2:
        label_map[sorted_clusters[0]] = "Early-N"
        label_map[sorted_clusters[1]] = "Late-N"
    elif n == 3:
        label_map[sorted_clusters[0]] = "Early-N"
        label_map[sorted_clusters[1]] = "Mid-N"
        label_map[sorted_clusters[2]] = "Late-N"
    else:
        for i, c in enumerate(sorted_clusters):
            if i == 0:
                label_map[c] = "Early-N"
            elif i == n - 1:
                label_map[c] = "Late-N"
            else:
                label_map[c] = f"Mid-N_{i}"

    return label_map


# ======================== Plotting + CSV Export Functions ==================

def plot_and_export_mean_profiles(
        profiles: np.ndarray, cluster_labels: np.ndarray,
        cluster_names: dict, target_len: int,
        outfile_plot: str, outfile_csv: str):
    """
    Line plot of mean ± SEM importance profiles per cluster.
    Also exports the underlying data as CSV.
    """
    n_clusters = len(set(cluster_labels))
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4),
                             squeeze=False)
    axes = axes[0]
    x_pos = np.linspace(1, target_len, target_len)
    palette = sns.color_palette("Set2", n_clusters)

    csv_rows = []

    for ax_i, (cid, cname) in enumerate(sorted(cluster_names.items())):
        ax = axes[ax_i]
        mask = cluster_labels == cid
        cl_profs = profiles[mask]
        mean_p = cl_profs.mean(axis=0)
        sem_p = cl_profs.std(axis=0) / np.sqrt(max(cl_profs.shape[0], 1))

        ax.fill_between(x_pos, mean_p - sem_p, mean_p + sem_p,
                        alpha=0.3, color=palette[ax_i])
        ax.plot(x_pos, mean_p, color=palette[ax_i], lw=2)
        ax.set_title(f"{cname}\n(n={cl_profs.shape[0]})")
        ax.set_xlabel("N-terminal position (resampled)")
        ax.set_ylabel("Normalised importance")

        for j in range(target_len):
            csv_rows.append({
                "Cluster": cname,
                "Resampled_Position": j + 1,
                "Mean_Importance": mean_p[j],
                "SEM": sem_p[j],
                "N_Proteins": cl_profs.shape[0]
            })

    plt.suptitle("Mean N‑terminal importance profiles by cluster")
    plt.tight_layout()
    plt.savefig(outfile_plot, dpi=300)
    plt.close()
    print(f"  Profile plot saved to: {outfile_plot}")

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(outfile_csv, index=False)
    print(f"  Profile data table saved to: {outfile_csv}")


def plot_and_export_heatmap(
        profiles: np.ndarray, cluster_labels: np.ndarray,
        cluster_names: dict, gene_names: list,
        target_len: int,
        outfile_plot: str, outfile_csv: str):
    """
    Heatmap of all importance profiles sorted by cluster.
    Cluster labels are placed in a dedicated axis strip to the right
    of the heatmap (outside the main plotting area).
    Also exports the resampled profile matrix as CSV.
    """
    # Sort: first by cluster, then by peak position within each cluster
    order = np.lexsort(([find_peak_position(p) for p in profiles],
                        cluster_labels))
    sorted_profs = profiles[order]
    sorted_clusters = cluster_labels[order]
    sorted_genes = [gene_names[i] for i in order]

    # ---- figure and main heatmap axes ----
    fig, ax = plt.subplots(figsize=(14, max(8, len(gene_names) * 0.12)))

    im = ax.imshow(sorted_profs, aspect='auto', cmap='inferno',
                   interpolation='bilinear')
    ax.set_xlabel("N-terminal position (resampled)")
    ax.set_ylabel("Protein")

    # Cluster boundary lines
    prev_c = sorted_clusters[0]
    for i, c in enumerate(sorted_clusters):
        if c != prev_c:
            ax.axhline(i - 0.5, color='cyan', lw=1.5)
            prev_c = c

    # ---- right-hand strip for cluster labels ----
    divider = make_axes_locatable(ax)

    # Narrow axis for cluster-name labels
    ax_label = divider.append_axes("right", size="4%", pad="2%")
    y_ticks, y_labels = [], []
    for cid in sorted(set(sorted_clusters)):
        pos = np.where(sorted_clusters == cid)[0]
        y_ticks.append(pos.mean())
        y_labels.append(cluster_names.get(cid, f"Cluster_{cid}"))
    ax_label.set_yticks(y_ticks)
    ax_label.set_yticklabels(y_labels, fontsize=9)
    ax_label.set_ylim(ax.get_ylim())
    # Remove all spines / ticks from the label axis except right-side labels
    ax_label.tick_params(left=False, right=False,
                         labelleft=False, labelright=True)
    for spine in ax_label.spines.values():
        spine.set_visible(False)

    # Colorbar in its own narrow axis further right
    cax = divider.append_axes("right", size="2%", pad="3%")
    plt.colorbar(im, cax=cax, label="Normalised importance")

    plt.title("N‑terminal importance profiles (sorted by cluster)")
    plt.tight_layout()
    plt.savefig(outfile_plot, dpi=300)
    plt.close()
    print(f"  Heatmap saved to: {outfile_plot}")

    # ---- CSV export ----
    csv_data = {"Gene": sorted_genes,
                "Cluster": [cluster_names.get(c, f"Cluster_{c}")
                            for c in sorted_clusters]}
    for j in range(target_len):
        csv_data[f"Pos_{j+1}"] = sorted_profs[:, j]
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(outfile_csv, index=False)
    print(f"  Heatmap data table saved to: {outfile_csv}")


def plot_and_export_peak_distribution(
        profiles: np.ndarray, cluster_labels: np.ndarray,
        cluster_names: dict, gene_names: list,
        outfile_plot: str, outfile_csv: str):
    """
    Histogram of peak positions per cluster.
    Also exports per-gene peak positions as CSV.
    """
    n_clusters = len(set(cluster_labels))
    fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 1.5 * n_clusters),
                             squeeze=False, sharex=True)
    axes = axes[:, 0]

    csv_rows = []

    for ax_i, (cid, cname) in enumerate(sorted(cluster_names.items())):
        ax = axes[ax_i]
        mask = cluster_labels == cid
        peaks = [find_peak_position(p) for p in profiles[mask]]
        ax.hist(peaks, bins=20, alpha=0.6, color='steelblue', edgecolor='k')
        ax.axvline(np.mean(peaks), color='red', linestyle='--',
                   label=f"mean = {np.mean(peaks):.2f}")
        ax.set_ylabel(cname)
        ax.legend(fontsize=8)

        for gene, peak in zip(np.array(gene_names)[mask], peaks):
            csv_rows.append({
                "Gene": gene,
                "Cluster": cname,
                "Peak_Relative_Position": peak
            })

    axes[-1].set_xlabel(
        "Relative peak position (0 = N‑terminus, 1 = window end)")
    plt.suptitle("Peak position distribution by cluster")
    plt.tight_layout()
    plt.savefig(outfile_plot, dpi=300)
    plt.close()
    print(f"  Peak distribution saved to: {outfile_plot}")

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(outfile_csv, index=False)
    print(f"  Peak position data table saved to: {outfile_csv}")


def plot_and_export_population_mean(
        data: dict, nterm_driven_genes: list, weak_genes: list,
        nterm_lengths: dict, output_dir: str):
    """
    Population-averaged importance profile (first 50% of sequence),
    with N-term-driven vs Weak-N overlay.
    Exports CSV with per-position mean importance overall and by group.
    """
    max_pos = max(len(df) for df in data.values())
    acc_all, cnt_all = np.zeros(max_pos), np.zeros(max_pos)
    acc_driven, cnt_driven = np.zeros(max_pos), np.zeros(max_pos)
    acc_weak, cnt_weak = np.zeros(max_pos), np.zeros(max_pos)

    for gene, df in data.items():
        pos = df["position"].values.astype(int) - 1
        imp = df["importance"].values.astype(float)
        for p, v in zip(pos, imp):
            if p < max_pos:
                acc_all[p] += v
                cnt_all[p] += 1
                if gene in nterm_driven_genes:
                    acc_driven[p] += v
                    cnt_driven[p] += 1
                if gene in weak_genes:
                    acc_weak[p] += v
                    cnt_weak[p] += 1

    mean_all = np.divide(acc_all, cnt_all,
                         out=np.zeros_like(acc_all), where=cnt_all > 0)
    mean_driven = np.divide(acc_driven, cnt_driven,
                            out=np.zeros_like(acc_driven),
                            where=cnt_driven > 0)
    mean_weak = np.divide(acc_weak, cnt_weak,
                          out=np.zeros_like(acc_weak), where=cnt_weak > 0)

    n_show = int(max_pos * 0.5)
    n_show = min(n_show, max_pos)

    fig, ax = plt.subplots(figsize=(14, 4))
    x_ax = range(1, n_show + 1)
    ax.plot(x_ax, mean_all[:n_show], color='black', lw=1.5,
            label='All proteins')
    ax.plot(x_ax, mean_driven[:n_show], color='steelblue', lw=1,
            label='N‑terminal‑driven')
    ax.plot(x_ax, mean_weak[:n_show], color='salmon', lw=1,
            label='Weak‑N dependency')
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean importance")
    ax.set_title("Population‑averaged importance profile "
                 "(first 50% of sequence)")
    ax.legend()
    plt.tight_layout()

    outfile_plot = Path(output_dir) / "population_mean_profile.png"
    plt.savefig(outfile_plot, dpi=300)
    plt.close()
    print(f"  Population mean profile saved to: {outfile_plot}")

    csv_data = []
    for p in range(n_show):
        csv_data.append({
            "Position": p + 1,
            "Mean_Importance_All": mean_all[p],
            "Count_All": int(cnt_all[p]),
            "Mean_Importance_NtermDriven": mean_driven[p],
            "Count_NtermDriven": int(cnt_driven[p]),
            "Mean_Importance_WeakN": mean_weak[p],
            "Count_WeakN": int(cnt_weak[p]),
        })

    outfile_csv = Path(output_dir) / "population_mean_profile.csv"
    pd.DataFrame(csv_data).to_csv(outfile_csv, index=False)
    print(f"  Population mean data table saved to: {outfile_csv}")


def export_cluster_gene_lists(cluster_assignments: dict, output_dir: str,
                              cluster_type_name: str = "Nterm"):
    """
    Export per-cluster gene lists in the format:
        Cluster, Gene_Count, Gene_List
    """
    cluster_to_genes = {}
    for gene, cname in cluster_assignments.items():
        cluster_to_genes.setdefault(cname, []).append(gene)

    rows = []
    for cname in sorted(cluster_to_genes.keys()):
        genes = sorted(cluster_to_genes[cname])
        rows.append({
            "Cluster": cname,
            "Gene_Count": len(genes),
            "Gene_List": "; ".join(genes)
        })

    df = pd.DataFrame(rows)
    prefix = cluster_type_name.lower().replace(" ", "_")
    outpath = Path(output_dir) / f"{prefix}_genes.csv"
    df.to_csv(outpath, index=False)
    print(f"  Gene list saved to: {outpath}")
    return df


# ======================== Main Pipeline ===================================

def run_clustering(data: dict, nterm_proportion: float = 0.15,
                   nterm_min: int = 30, nterm_max: int = 100,
                   weak_threshold: float = 1.0,
                   peak_ratio: float = 0.85,
                   peak_factor: float = 1.5,
                   mode: str = "diva",
                   n_clusters: int = None,
                   output_dir: str = "./nterm_clusters"):
    """
    Run the complete N-terminal importance clustering pipeline.

    Parameters
    ----------
    data : dict
    nterm_proportion : float
    nterm_min, nterm_max : int
    weak_threshold : float
    peak_ratio : float   [diva mode]
    peak_factor : float  [pivot mode]
    mode : str
    n_clusters : int or None
    output_dir : str
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = sorted(data.keys())
    n_total = len(genes)

    # Build a compact mode-description for the log line
    if mode == "diva":
        mode_desc = f"peak_ratio={peak_ratio}"
    else:
        mode_desc = f"peak_factor={peak_factor}"

    # ------------------------------------------------------------------
    # Step 1: Determine N‑terminal window & classify each protein
    # ------------------------------------------------------------------
    print(f"\nStep 1: Classifying {n_total} proteins "
          f"(mode={mode}, {mode_desc}, weak_threshold={weak_threshold})...")
    nterm_driven_genes = []
    weak_genes = []
    nterm_profiles_raw = {}
    nterm_lengths = {}

    for gene in genes:
        df = data[gene]
        prot_len = len(df)
        nlen = compute_nterm_length(prot_len, nterm_proportion,
                                    nterm_min, nterm_max)
        nterm_lengths[gene] = nlen

        if is_nterm_driven(df, nlen, weak_threshold,
                           peak_ratio, peak_factor, mode):
            nterm_driven_genes.append(gene)
            nterm_profiles_raw[gene] = extract_nterm_profile(df, nlen)
        else:
            weak_genes.append(gene)

    n_driven = len(nterm_driven_genes)
    n_weak = len(weak_genes)
    print(f"  N-terminal driven: {n_driven} proteins")
    print(f"  Weak N-terminal dependency: {n_weak} proteins")

    # ------------------------------------------------------------------
    # Step 2: Resample N‑terminal profiles to fixed length
    # ------------------------------------------------------------------
    TARGET_LEN = 50
    flipped_list = []
    if n_driven >= 2:
        print(f"\nStep 2: Resampling {n_driven} N-terminal profiles "
              f"to {TARGET_LEN} points...")
        resampled_list = []
        for g in nterm_driven_genes:
            prof, was_flipped = resample_profile(
                nterm_profiles_raw[g], TARGET_LEN, enforce_direction=False)
            resampled_list.append(prof)
            flipped_list.append(was_flipped)
        resampled = np.array(resampled_list)
        n_flipped = sum(flipped_list)
        if n_flipped > 0:
            print(f"  Direction correction: {n_flipped} profiles flipped "
                  f"to align dominant peak in first half.")
    else:
        resampled = np.empty((0, TARGET_LEN))

    # ------------------------------------------------------------------
    # Step 3: Hierarchical clustering of resampled profiles
    # ------------------------------------------------------------------
    cluster_labels_driven = np.array([], dtype=int)
    cluster_names = {}
    if n_driven >= 3:
        print(f"\nStep 3: Clustering N-terminal-driven proteins...")

        dist_vec = pdist(resampled, metric='correlation')
        dist_vec = np.nan_to_num(dist_vec, nan=0.0)
        Z = linkage(dist_vec, method='average')

        if n_clusters is None:
            merge_dists = Z[:, 2]
            if len(merge_dists) >= 2:
                jumps = np.diff(merge_dists)
                best_idx = np.argmax(jumps[-min(10, len(jumps)):])
                best_idx = len(jumps) - min(10, len(jumps)) + best_idx
                n_clusters = len(merge_dists) - best_idx
                n_clusters = max(3, min(6, n_clusters))
            else:
                n_clusters = 3
        print(f"  Number of clusters: {n_clusters}")

        cluster_labels_driven = fcluster(Z, n_clusters,
                                         criterion='maxclust')
        cluster_names = label_clusters(resampled, cluster_labels_driven)

        for cid in sorted(set(cluster_labels_driven)):
            cname = cluster_names.get(cid, f"Cluster_{cid}")
            n_members = np.sum(cluster_labels_driven == cid)
            peak_std = np.std(
                [find_peak_position(p)
                 for p in resampled[cluster_labels_driven == cid]])
            print(f"    {cname}: {n_members} proteins "
                  f"(peak std={peak_std:.3f})")

    elif n_driven == 2:
        cluster_labels_driven = np.array([0, 1])
        cluster_names = {0: "Profile_A", 1: "Profile_B"}
        print(f"  Only 2 driven proteins — assigned to individual profiles.")
    else:
        print(f"  Too few N-terminal-driven proteins for clustering.")

    # ------------------------------------------------------------------
    # Step 4: Build final assignment table
    # ------------------------------------------------------------------
    assignments = {}
    for i, gene in enumerate(nterm_driven_genes):
        cid = (cluster_labels_driven[i]
               if len(cluster_labels_driven) > 0 else -1)
        cname = cluster_names.get(cid, f"Cluster_{cid}")
        assignments[gene] = cname

    for gene in weak_genes:
        assignments[gene] = "Weak-N"

    assign_df = pd.DataFrame([
        {"Gene": g, "Nterm_Profile_Type": assignments[g],
         "Nterm_Window_Length": nterm_lengths[g]}
        for g in genes
    ])
    assign_path = outdir / "nterm_profile_assignments.csv"
    assign_df.to_csv(assign_path, index=False)
    print(f"\n  Assignments saved to: {assign_path}")

    # ------------------------------------------------------------------
    # Step 5: Export per‑cluster gene lists
    # ------------------------------------------------------------------
    export_cluster_gene_lists(assignments, output_dir, "Nterm_Profile")

    # ------------------------------------------------------------------
    # Step 6: Figures + data tables
    # ------------------------------------------------------------------
    print(f"\nStep 4: Generating figures and data tables...")

    if n_driven >= 2:
        plot_and_export_mean_profiles(
            resampled, cluster_labels_driven, cluster_names, TARGET_LEN,
            str(outdir / "nterm_cluster_profiles.png"),
            str(outdir / "nterm_cluster_profiles_data.csv"))

        plot_and_export_heatmap(
            resampled, cluster_labels_driven, cluster_names,
            nterm_driven_genes, TARGET_LEN,
            str(outdir / "nterm_profile_heatmap.png"),
            str(outdir / "nterm_profile_heatmap_data.csv"))

        plot_and_export_peak_distribution(
            resampled, cluster_labels_driven, cluster_names,
            nterm_driven_genes,
            str(outdir / "nterm_peak_distribution.png"),
            str(outdir / "nterm_peak_distribution_data.csv"))

    plot_and_export_population_mean(
        data, set(nterm_driven_genes), set(weak_genes),
        nterm_lengths, output_dir)

    # ------------------------------------------------------------------
    # Step 7: Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Clustering summary:")
    print(f"  Total proteins:            {n_total}")
    print(f"  N‑terminal‑driven:         {n_driven} "
          f"({100*n_driven/n_total:.1f}%)")
    for cid in sorted(set(cluster_labels_driven)):
        cname = cluster_names.get(cid, f"Cluster_{cid}")
        n_m = np.sum(cluster_labels_driven == cid)
        peak_std = np.std(
            [find_peak_position(p)
             for p in resampled[cluster_labels_driven == cid]])
        print(f"    ↳ {cname}:  {n_m} proteins (peak std={peak_std:.3f})")
    print(f"  Weak N‑terminal dependency: {n_weak} "
          f"({100*n_weak/n_total:.1f}%)")
    if sum(flipped_list) > 0:
        print(f"  Direction-corrected profiles: {sum(flipped_list)}")
    print(f"{'='*50}")

    return assignments


# ======================== CLI Entry Point =================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster P450 proteins by N-terminal importance profiles")
    p.add_argument("--importance-dir", required=True,
                   help="Folder containing *_full_importance.csv "
                        "(PIVOT/SVM) or *_residue_importance.csv (DIVA) files")
    p.add_argument("--output-dir", default="./nterm_clusters",
                   help="Output directory (default: ./nterm_clusters)")
    p.add_argument("--nterm-proportion", type=float, default=0.15,
                   help="Fraction of protein length defining N‑terminal "
                        "window (default: 0.15)")
    p.add_argument("--nterm-min", type=int, default=30,
                   help="Minimum N‑terminal window length (default: 30)")
    p.add_argument("--nterm-max", type=int, default=100,
                   help="Maximum N‑terminal window length (default: 100)")
    p.add_argument("--weak-threshold", type=float, default=1.0,
                   help="Mean(N-term)/Mean(whole) ratio below which a "
                        "protein is classified as weak-N (default: 1.0)")
    p.add_argument("--mode", default="diva", choices=["diva", "pivot"],
                   help="Peak criterion mode.  'diva': max(N-term) >= "
                        "peak_ratio * max(whole); 'pivot': max(N-term) >= "
                        "peak_factor * mean(N-term).  (default: diva)")
    p.add_argument("--peak-ratio", type=float, default=0.85,
                   help="[diva mode] Fraction of global max that N‑term "
                        "max must reach.  Lower → more N‑terminal‑driven.  "
                        "Recommended range 0.75–0.90.  (default: 0.85)")
    p.add_argument("--peak-factor", type=float, default=1.5,
                   help="[pivot mode] Factor by which N‑term max must "
                        "exceed N‑term mean for a peak to be recognised.  "
                        "(default: 1.5)")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Number of clusters for N-term-driven group "
                        "(default: auto, minimum 3)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("Loading importance profiles...")
    data = load_importance_files(args.importance_dir)

    run_clustering(
        data,
        nterm_proportion=args.nterm_proportion,
        nterm_min=args.nterm_min,
        nterm_max=args.nterm_max,
        weak_threshold=args.weak_threshold,
        peak_ratio=args.peak_ratio,
        peak_factor=args.peak_factor,
        mode=args.mode,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
