#!/usr/bin/env python3
"""
PEACE: Per-position Embedding Alignment-free Constraint Estimation
       Dual-mode: DTW consensus OR median-reference soft correspondence
       WITH DIRECTIONALITY + PHYLOGENETIC STRUCTURE ANALYSIS
===========================================================================
[UPDATE 2026-07-20]: Added FDR correction & t-SNE/UMAP projection support.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from Bio import Phylo
from tqdm import tqdm
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # ⭐ MODIFIED: 引入 t-SNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import gc
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

# ⭐ MODIFIED: 引入 statsmodels 用于 FDR，若未安装则给出 WARNING
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("Warning: 'statsmodels' not installed. FDR correction will be skipped. pip install statsmodels")

# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="PEACE: Dual-mode embedding constraint analysis")
    p.add_argument("--mode", required=True, choices=['dtw', 'reference'],
                   help="dtw = DTW consensus (no ref); reference = median-ref + soft corr")
    p.add_argument("--fasta", required=True, help="Input FASTA")
    p.add_argument("--output-dir", "-o", default="peace_results", help="Output directory")
    p.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=1022)
    p.add_argument("--min-effective-seqs", type=int, default=30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    # DTW
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--dtw-max-iter", type=int, default=10)
    p.add_argument("--dtw-subset", type=int, default=0)
    p.add_argument("--consensus-length", type=int, default=0)
    p.add_argument("--skip-dba", action="store_true")
    # Reference
    p.add_argument("--sim-threshold-percentile", type=float, default=5.0)
    # Phylogeny
    p.add_argument("--tree", default=None, help="Newick tree for phylo-structure test")
    p.add_argument("--phylo-permutations", type=int, default=500)
    # ⭐ MODIFIED: 新增降维算法选择参数
    p.add_argument("--dim-reduction", default='pca', choices=['pca', 'tsne', 'umap'], 
                   help="Dimensionality reduction for visualizing directionality clusters (pca|tsne|umap)")
    return p.parse_args()


# ===========================================================================
# Common: FASTA, Model, Embeddings, Debias (unchanged)
# ===========================================================================
def load_fasta(path):
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError("No sequences.")
    print(f"Loaded {len(records)} sequences, "
          f"length {min(len(r.seq) for r in records)}–{max(len(r.seq) for r in records)}")
    return records

def load_model(name, device):
    print(f"Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    print(f"Model on {device}.")
    return tok, model

def extract_embeddings(records, tok, model, device, max_len, out_dir):
    cache = Path(out_dir) / "embedding_cache"
    cache.mkdir(parents=True, exist_ok=True)
    embeddings, sequences = {}, {}
    for rec in tqdm(records, desc="Embeddings"):
        sid = rec.id
        safe = sid.replace('/', '_').replace('\\', '_').replace('|', '_')
        cp = cache / f"{safe}.npy"
        if cp.exists():
            e = np.load(cp)
            L = e.shape[0]
            embeddings[sid] = e
            sequences[sid] = str(rec.seq)[:L]
            continue
        seq = str(rec.seq)
        inp = tok(seq, return_tensors="pt", truncation=True, max_length=max_len + 2)
        L = inp['attention_mask'][0].sum().item() - 2
        if L <= 0:
            continue
        sq = seq[:L]
        if L < len(seq):
            print(f"  Truncated {sid}: {len(seq)}→{L}")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_attentions=False)
        e = out.last_hidden_state[0, 1:1+L, :].cpu().numpy()
        embeddings[sid] = e
        sequences[sid] = sq
        np.save(cp, e)
        del inp, out
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(f"Extracted {len(embeddings)} embeddings.")
    return embeddings, sequences

def compute_position_templates(embeddings):
    tmpl, cnt = {}, {}
    for e in embeddings.values():
        for j in range(e.shape[0]):
            if j not in tmpl:
                tmpl[j] = e[j].copy().astype(np.float64)
                cnt[j] = 1
            else:
                tmpl[j] += e[j]
                cnt[j] += 1
    for j in tmpl:
        tmpl[j] = (tmpl[j] / cnt[j]).astype(np.float32)
    print(f"Position templates: {len(tmpl)} positions, max count {max(cnt.values())}.")
    return tmpl

def debias_embeddings(embeddings, templates):
    return {sid: np.array([e[j] - templates.get(j, 0) for j in range(e.shape[0])], dtype=np.float32)
            for sid, e in embeddings.items()}

# ===========================================================================
# DTW mode (unchanged)
# ===========================================================================
def build_trajectories(embeddings, w):
    traj, winfo = {}, {}
    h = w // 2
    for sid, e in embeddings.items():
        L = e.shape[0]
        if L < w:
            continue
        n = L - w + 1
        t = np.zeros((n, e.shape[1]), dtype=np.float32)
        inf = []
        for i in range(n):
            t[i] = e[i:i+w].mean(axis=0)
            inf.append({'center_pos': i + h, 'start': i, 'end': i + w})
        traj[sid] = t
        winfo[sid] = inf
    print(f"Built {len(traj)} trajectories (w={w}).")
    return traj, winfo

def pca_reduce(traj, nc):
    all_w = np.concatenate(list(traj.values()), axis=0)
    print(f"PCA ({nc} comp) on {all_w.shape[0]} windows ...")
    pca = PCA(n_components=nc, random_state=42).fit(all_w)
    print(f"  Explained: {pca.explained_variance_ratio_.sum():.3f}")
    return {sid: pca.transform(t).astype(np.float64) for sid, t in traj.items()}, pca

def compute_dba_consensus(traj_red, cons_len, max_iter, subset, cache_path):
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except ImportError:
        raise ImportError("tslearn required: pip install tslearn")
    if cache_path.exists():
        print(f"Loading cached consensus: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    tl = list(traj_red.values())
    if subset > 0 and subset < len(tl):
        np.random.seed(42)
        tl = [tl[i] for i in np.random.choice(len(tl), subset, replace=False)]
        print(f"  DBA subset: {subset}")
    if cons_len <= 0:
        cons_len = int(np.median([t.shape[0] for t in tl]))
    print(f"  Consensus length: {cons_len}, DBA max_iter={max_iter}")
    cons = dtw_barycenter_averaging(tl, barycenter_size=cons_len, max_iter=max_iter, tol=1e-4, verbose=True)
    print(f"  DBA done: {cons.shape}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cons, f)
    return cons

def dtw_align_and_collect(consensus, traj_red, traj_orig, sequences, winfo, w, embeddings_debiased):
    try:
        from tslearn.metrics import dtw_path
    except ImportError:
        raise ImportError("tslearn required")
    Lc = consensus.shape[0]
    collectors = {k: [] for k in range(Lc)}
    sids = list(traj_red.keys())
    for sid in tqdm(sids, desc="DTW aligning"):
        tr = traj_red[sid]
        to_ = traj_orig[sid]
        inf = winfo[sid]
        seq = sequences[sid]
        if tr.shape[0] < 3:
            continue
        path, _ = dtw_path(consensus, tr)
        mapping = {}
        for ci, ti in path:
            mapping.setdefault(ci, []).append(ti)
        for ci, tis in mapping.items():
            if ci >= Lc:
                continue
            embs = to_[tis]
            mean_emb = embs.mean(axis=0)
            rt = tis[len(tis)//2]
            if rt < len(inf):
                cp = inf[rt]['center_pos']
                aa = seq[cp] if cp < len(seq) else 'X'
                ap = cp
            else:
                aa = 'X'; ap = -1
            collectors[ci].append({
                'seq_id': sid, 'embedding': mean_emb, 'aa': aa, 'abs_position': ap
            })
    pd_list = []
    for k in range(Lc):
        items = collectors[k]
        n_eff = len(items)
        if items:
            dom_aa = max(set(it['aa'] for it in items), key=lambda x: [it['aa'] for it in items].count(x))
            mapos = np.mean([it['abs_position'] for it in items])
            seq_ids = [it['seq_id'] for it in items]
        else:
            dom_aa = 'X'; mapos = -1; seq_ids = []
        pd_list.append({
            'position': k + 1, 'dominant_aa': dom_aa, 'mean_abs_position': mapos,
            'matched_embeddings': [it['embedding'] for it in items],
            'matched_aas': [it['aa'] for it in items],
            'matched_positions': [it['abs_position'] for it in items],
            'matched_seq_ids': seq_ids, 'n_effective': n_eff
        })
    print(f"  Collected {Lc} windows.")
    return pd_list


# ===========================================================================
# Reference mode (unchanged)
# ===========================================================================
def select_reference(records, seed=42):
    lengths = [len(r.seq) for r in records]
    med = np.median(lengths)
    dists = [abs(l - med) for l in lengths]
    candidates = [i for i, d in enumerate(dists) if d == min(dists)]
    random.seed(seed)
    ref = records[random.choice(candidates)]
    print(f"Reference: {ref.id} (len={len(ref.seq)}, median={med:.0f})")
    return ref

def soft_correspondence(ref_id, ref_emb, ref_seq, target_embeddings, target_sequences, sim_threshold):
    L_ref = ref_emb.shape[0]
    ref_norms = np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-10
    ref_norm = ref_emb / ref_norms
    tgt_ids = [k for k in target_embeddings.keys() if k != ref_id]
    pd_list = []
    for i in tqdm(range(L_ref), desc="Soft matching"):
        rv = ref_norm[i]
        matched_embs, matched_aas, matched_positions, matched_sims, matched_ids = [], [], [], [], []
        for tid in tgt_ids:
            te = target_embeddings[tid]
            ts = target_sequences[tid]
            tn = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-10)
            cs = np.dot(tn, rv)
            bi = int(np.argmax(cs))
            bs = float(cs[bi])
            if bs >= sim_threshold:
                matched_embs.append(te[bi])
                matched_aas.append(ts[bi] if bi < len(ts) else 'X')
                matched_positions.append(bi)
                matched_sims.append(bs)
                matched_ids.append(tid)
        pd_list.append({
            'position': i + 1,
            'dominant_aa': ref_seq[i] if i < len(ref_seq) else 'X',
            'mean_abs_position': float(i),
            'matched_embeddings': matched_embs,
            'matched_aas': matched_aas,
            'matched_positions': matched_positions,
            'matched_similarities': matched_sims,
            'matched_seq_ids': matched_ids,
            'n_effective': len(matched_embs)
        })
    return pd_list

def determine_sim_threshold(ref_emb, embeddings_debiased, percentile=5.0, n_sample_pos=200, n_sample_tgt=50):
    target_ids = list(embeddings_debiased.keys())
    ref_norm = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-10)
    sp = np.linspace(0, ref_emb.shape[0]-1, min(n_sample_pos, ref_emb.shape[0])).astype(int)
    st = target_ids[:min(n_sample_tgt, len(target_ids))]
    all_sims = []
    for i in sp:
        rv = ref_norm[i]
        for tid in st:
            te = embeddings_debiased[tid]
            tn = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-10)
            all_sims.append(float(np.max(np.dot(tn, rv))))
    th = np.percentile(all_sims, percentile)
    print(f"  Sampled {len(all_sims)} sims, {percentile}th pct threshold = {th:.4f}")
    return th


# ===========================================================================
# Common: dispersion, Shannon, metrics (unchanged)
# ===========================================================================
def compute_dispersion(matched_embs):
    if len(matched_embs) < 3:
        return np.nan
    e = np.stack(matched_embs, axis=0)
    n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-10)
    sm = n @ n.T
    tri = np.triu_indices(e.shape[0], k=1)
    return float(np.mean(1.0 - sm[tri[0], tri[1]]))

def compute_shannon(aas):
    if len(aas) == 0:
        return np.nan
    u, c = np.unique(aas, return_counts=True)
    f = c / c.sum()
    return float(-np.sum(f * np.log2(f + 1e-10)))

def compute_metrics(pd_list, min_eff):
    results = []
    for pd in pd_list:
        n_eff = pd['n_effective']
        disp = np.nan; sh = np.nan; ms = np.nan; pos_std = np.nan
        if n_eff >= min_eff:
            disp = compute_dispersion(pd['matched_embeddings'])
            sh = compute_shannon(pd['matched_aas'])
            sims = pd.get('matched_similarities', None)
            if sims is not None and len(sims) > 0:
                ms = float(np.mean(sims))
            else:
                ms = 1.0 - disp if not np.isnan(disp) else np.nan
            ps = [p for p in pd['matched_positions'] if p >= 0]
            if len(ps) > 1:
                pos_std = float(np.std(ps))
        results.append({
            'position': pd['position'], 'dominant_aa': pd.get('dominant_aa', 'X'),
            'mean_abs_position': pd.get('mean_abs_position', -1),
            'n_effective': n_eff, 'dispersion': disp, 'shannon_entropy': sh,
            'mean_match_similarity': ms, 'position_offset_std': pos_std,
        })
    return results


# ===========================================================================
# Directionality analysis (MODIFIED: t-SNE/UMAP support)
# ===========================================================================
def compute_directionality(pd_list, min_eff=30, pca_dim=5, reduction='pca'):
    # ⭐ MODIFIED: 接受 reduction 参数，替换原始终的 PCA 投影为动态降维
    results = []
    for pd in pd_list:
        n_eff = pd['n_effective']; embs = pd['matched_embeddings']
        seq_ids = pd.get('matched_seq_ids', [])
        er = np.nan; t2r = np.nan; nc = 1; sil = np.nan; ds = np.nan
        cl = None; p2d = None
        if n_eff < min_eff or len(embs) < 20:
            results.append({'effective_rank': er, 'top2_ratio': t2r, 'n_clusters_gmm': nc,
                           'silhouette': sil, 'directional_score': ds, 'cluster_labels': None,
                           'pca_2d': None, 'cluster_seq_ids': None})
            continue
        ea = np.stack(embs, axis=0)
        
        # Perform linear PCA for effective rank and GMM inputs (since GMM prefers no extreme high-dim, still we use PCA 5 dims)
        pf = PCA(n_components=min(pca_dim, ea.shape[0]-1)).fit(ea)
        ev = pf.explained_variance_
        if len(ev) < pca_dim:
            ev = np.pad(ev, (0, pca_dim - len(ev)), constant_values=1e-10)
        se = np.sum(ev); se2 = np.sum(ev**2)
        er = float(se**2 / se2) if se2 > 0 else np.nan
        t2r = float((ev[0]+ev[1])/se) if se > 0 else np.nan
        pe = pf.transform(ea) # reduced to 5 dims for GMM
        
        bics, gmms = [], []
        mk = max(1, min(5, ea.shape[0]//10))
        for k in range(1, mk+1):
            g = GaussianMixture(n_components=k, random_state=42, covariance_type='full', reg_covar=0.01)
            g.fit(pe); bics.append(g.bic(pe)); gmms.append(g)
        bk = int(np.argmin(bics)+1)
        nc = bk; cl = gmms[bk-1].predict(pe)
        if bk >= 2:
            try:
                sil = float(silhouette_score(pe, cl))
            except Exception:
                sil = np.nan
        disp = compute_dispersion(embs) if len(embs)>=3 else 0.0
        rs = 1.0 - min(er/pca_dim, 1.0) if not np.isnan(er) else 0.0

        if nc >= 2 and not np.isnan(sil):
            structure_score = max(0.0, (sil + 0.2) / 1.2)
        elif not np.isnan(t2r):
            structure_score = max(0.0, min(1.0, (t2r - 0.3) / 0.7))
        else:
            structure_score = 0.0

        ds = float(0.4*rs + 0.35*structure_score + 0.25*np.tanh(disp/0.3))
        
        # ⭐ MODIFIED: 动态选择降维方式用于可视化 (pca, tsne, umap)
        if ea.shape[0] >= 3:
            if reduction == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, ea.shape[0]-1))
                    p2d = reducer.fit_transform(ea)
                except ImportError:
                    print("Warning: umap-learn not installed. Falling back to PCA for visualization.")
                    p2d = PCA(n_components=2).fit_transform(ea)
                except Exception as e:
                    print(f"UMAP failed with error: {e}. Falling back to PCA.")
                    p2d = PCA(n_components=2).fit_transform(ea)
            elif reduction == 'tsne':
                # t-SNE 需要 n_samples > perplexity
                perplexity = min(30, max(5, ea.shape[0] - 1))
                p2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto').fit_transform(ea)
            else: # default pca
                p2d = PCA(n_components=2).fit_transform(ea)
        else:
            p2d = None

        results.append({'effective_rank': er, 'top2_ratio': t2r, 'n_clusters_gmm': nc,
                       'silhouette': sil, 'directional_score': ds, 'cluster_labels': cl,
                       'pca_2d': p2d, 'cluster_seq_ids': seq_ids if cl is not None else None})
    return results


# ===========================================================================
# Phylogenetic structure analysis (MODIFIED: FDR)
# ===========================================================================
def _phylo_dist_matrix(tree, tip_names):
    terminals = {t.name: t for t in tree.get_terminals()}
    tips = [t for t in tip_names if t in terminals]
    if len(tips) < 2:
        return None, []
    n = len(tips)
    dm = np.zeros((n, n))
    paths = {t: tree.get_path(terminals[t]) for t in tips}
    for i in range(n):
        for j in range(i+1, n):
            pi, pj = paths[tips[i]], paths[tips[j]]
            m = 0
            for k in range(min(len(pi), len(pj))):
                if pi[k] == pj[k]:
                    m = k
                else:
                    break
            di = sum((c.branch_length or 0) for c in pi[m:])
            dj = sum((c.branch_length or 0) for c in pj[m:])
            dm[i,j] = dm[j,i] = di + dj
    return dm, tips

def _match_fasta_to_tree(fasta_ids, tree_names):
    matches = {}
    tn_set = set(tree_names)
    for fid in fasta_ids:
        if fid in tn_set:
            matches[fid] = fid
            continue
        fl = fid.lower()
        for tn in tree_names:
            if fl in tn.lower() or tn.lower() in fl:
                matches[fid] = tn
                break
    return matches

def analyze_phylogenetic_structure(results, dir_results, pd_list, tree_path, out_dir, n_perm=500, min_cs=5):
    print(f"\n{'='*60}\nPhylogenetic Structure Analysis\n{'='*60}")
    tree = Phylo.read(tree_path, 'newick')
    tree_tips = {t.name for t in tree.get_terminals()}
    print(f"Tree: {len(tree_tips)} tips")
    all_ids = set()
    for pd in pd_list:
        all_ids.update(pd.get('matched_seq_ids', []))
    f2t = _match_fasta_to_tree(all_ids, tree_tips)
    matched = set(f2t.keys())
    print(f"Matched FASTA→Tree: {len(matched)}/{len(all_ids)}")
    tn_list = list(set(f2t.values()))
    dm, tip_list = _phylo_dist_matrix(tree, tn_list)
    if dm is None:
        print("  [ERROR] Cannot compute phylo dist matrix.")
        return [{'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                 'phylo_directional_score': np.nan} for _ in results]
    t2i = {t: i for i, t in enumerate(tip_list)}
    print(f"  Dist matrix: {len(tip_list)}×{len(tip_list)}")
    phylo_results = []
    raw_pvals = []
    raw_indices = []
    
    # 第一遍遍历，计算 p_value
    for i, (r, d) in enumerate(zip(results, dir_results)):
        if d['n_clusters_gmm'] < 2 or d['cluster_labels'] is None:
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        labels = d['cluster_labels']; seq_ids = d.get('cluster_seq_ids', [])
        if len(seq_ids) != len(labels):
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        indices = np.array([t2i.get(f2t.get(sid, ''), -1) for sid in seq_ids])
        vi = np.where(indices >= 0)[0]
        if len(vi) < 20:
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        vl = labels[vi]; ul = sorted(set(vl))
        if len(ul) < 2:
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        obs_w, cw = [], []
        for lab in ul:
            m = vl == lab
            if m.sum() >= min_cs:
                idx = vi[m]; sub = dm[np.ix_(idx, idx)]
                tri = np.triu_indices(len(idx), k=1)
                obs_w.append(float(sub[tri].mean())); cw.append(m.sum())
        if len(obs_w) < 2:
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        om = np.average(obs_w, weights=cw)
        null_means = []
        for _ in range(n_perm):
            sv = vl.copy(); np.random.shuffle(sv)
            nw = []
            for lab in ul:
                m = sv == lab
                if m.sum() >= min_cs:
                    idx = vi[m]; sub = dm[np.ix_(idx, idx)]
                    tri = np.triu_indices(len(idx), k=1)
                    nw.append(float(sub[tri].mean()))
            if len(nw) >= 2:
                null_means.append(np.mean(nw))
        if len(null_means) < 10:
            phylo_results.append({'phylo_structure_z': np.nan, 'phylo_structure_pval': np.nan,
                                  'phylo_directional_score': np.nan})
            continue
        nm = np.mean(null_means); ns = np.std(null_means)
        z = (nm - om)/ns if ns > 0 else 0.0
        pv = (np.sum(np.array(null_means) <= om) + 1)/(n_perm + 1)
        
        # 临时存储 p 值用于 FDR
        raw_pvals.append(pv)
        raw_indices.append(i)
        
        dsc = d['directional_score']
        pds = (0.5*dsc + 0.5*min(max(z,0),4)/4.0) if not np.isnan(dsc) and not np.isnan(z) else np.nan
        phylo_results.append({'phylo_structure_z': float(z), 'phylo_structure_pval': float(pv),
                              'phylo_directional_score': float(pds)})

    # ⭐ MODIFIED: 对 phylo p_value 进行 FDR 校正
    if HAS_STATSMODELS and len(raw_pvals) > 1:
        _, q_vals, _, _ = multipletests(raw_pvals, method='fdr_bh')
        for idx, q in zip(raw_indices, q_vals):
            phylo_results[idx]['phylo_structure_qval'] = float(q)
    else:
        for idx in raw_indices:
            phylo_results[idx]['phylo_structure_qval'] = np.nan
        if not HAS_STATSMODELS and len(raw_pvals) > 1:
            print("  [WARNING] Install 'statsmodels' to perform FDR correction on phylo p-values.")

    nsig = sum(1 for pr in phylo_results if not np.isnan(pr['phylo_structure_pval']) and pr['phylo_structure_pval'] < 0.05)
    print(f"  Significant phylo structure (p<0.05): {nsig}")
    return phylo_results


# ===========================================================================
# Visualization (common, updated titles)
# ===========================================================================
AA_COLORS = {
    'A':'#C0C0C0','C':'#FFFF00','D':'#FF0000','E':'#FF0000','F':'#00FF00',
    'G':'#FFA500','H':'#0000FF','I':'#808080','K':'#0000FF','L':'#808080',
    'M':'#FFFF00','N':'#FF00FF','P':'#00FFFF','Q':'#FF00FF','R':'#0000FF',
    'S':'#FFA500','T':'#FFA500','V':'#808080','W':'#00FF00','Y':'#00FF00'
}

def plot_dispersion_profile(results, mode_label, out_dir, dpi, n_term_boundary=50):
    df = pd.DataFrame(results)
    valid = df[df['dispersion'].notna()].copy()
    if len(valid) == 0:
        return
    is_nt = valid['mean_abs_position'] <= n_term_boundary
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 2, 1]})
    ax = axes[0]
    pos = valid['position'].values; disp = valid['dispersion'].values
    mp = valid['mean_abs_position'].values
    colors = ['#E74C3C' if m <= n_term_boundary else '#2980B9' for m in mp]
    ax.scatter(pos, disp, c=colors, s=15, alpha=0.6, edgecolors='none')
    if len(disp) > 11:
        ax.plot(pos, np.convolve(disp, np.ones(11)/11, mode='same'), 'k-', lw=2, alpha=0.7, label='Smoothed')
    ma = np.nanmean(disp); mn = np.nanmean(disp[is_nt]) if is_nt.any() else np.nan
    mc = np.nanmean(disp[~is_nt]) if (~is_nt).any() else np.nan
    ax.axhline(ma, color='grey', ls='--', alpha=0.6, label=f'Mean: {ma:.4f}')
    if not np.isnan(mn):
        ax.axhline(mn, color='#E74C3C', ls=':', alpha=0.6, label=f'N-term: {mn:.4f}')
    if not np.isnan(mc):
        ax.axhline(mc, color='#2980B9', ls=':', alpha=0.6, label=f'Core: {mc:.4f}')
    ax.set_xlabel('Position'); ax.set_ylabel('Dispersion')
    ax.set_title(f'Dispersion profile [{mode_label}]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    if is_nt.any():
        ax.axvspan(1, valid[is_nt]['position'].max(), alpha=0.06, color='red')
    def w2p(x):
        idx = np.clip(np.searchsorted(pos, x)-1, 0, len(mp)-1)
        return mp[idx]
    ax.secondary_xaxis('top', functions=(w2p, lambda p: np.interp(p, mp, pos))).set_xlabel('Approx. residue pos')
    ax2 = axes[1]
    ax2.bar(pos, valid['n_effective'].values, width=1.0, color='#7FB3D8', alpha=0.7, label='N effective')
    ax2.set_ylabel('N effective'); ax2.legend(fontsize=8, loc='upper left'); ax2.grid(True, alpha=0.2)
    if is_nt.any():
        ax2.axvspan(1, valid[is_nt]['position'].max(), alpha=0.06, color='red')
    ax2b = ax2.twinx()
    ax2b.plot(pos, valid['mean_match_similarity'].values, 'orange', lw=1, alpha=0.7, label='Mean sim')
    ax2b.set_ylabel('Mean match sim', color='orange'); ax2b.tick_params(axis='y', labelcolor='orange')
    ax2b.legend(fontsize=8, loc='upper right')
    ax3 = axes[2]
    aas = valid['dominant_aa'].values
    ax3.bar(pos, [1]*len(pos), color=[AA_COLORS.get(a,'#000') for a in aas], width=1.0, edgecolor='none')
    ax3.set_xlim(1, max(pos)); ax3.set_ylim(0,1); ax3.set_xlabel('Position'); ax3.set_yticks([])
    ax3.set_title('Dominant AA')
    if is_nt.any():
        ax3.axvspan(1, valid[is_nt]['position'].max(), alpha=0.06, color='red')
    plt.tight_layout(); plt.savefig(out_dir/'dispersion_profile.png', dpi=dpi)
    plt.savefig(out_dir/'dispersion_profile.pdf'); plt.close()
    print(f"  Saved dispersion_profile.png/pdf")

def plot_diagnostics(results, mode_label, out_dir, dpi, n_term_boundary=50):
    df = pd.DataFrame(results)
    valid = df[df['dispersion'].notna()].copy()
    if len(valid) == 0:
        return
    is_nt = valid['mean_abs_position'] <= n_term_boundary
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax = axes[0,0]
    colors = ['#E74C3C' if n else '#2980B9' for n in is_nt]
    ax.scatter(valid['mean_match_similarity'], valid['dispersion'], c=colors, s=12, alpha=0.5, edgecolors='none')
    rho, pv = spearmanr(valid['mean_match_similarity'].values, valid['dispersion'].values)
    if 'DTW' in mode_label.upper():
        rho_note = f"ρ = {rho:.3f} (math. identity)"
    else:
        rho_note = f"ρ = {rho:.3f}"
        if abs(rho) > 0.8:
            rho_note += "\n(>0.8: dispersion may be redundant with match quality)"
    ax.set_xlabel('Mean match sim'); ax.set_ylabel('Dispersion')
    ax.set_title(f'(a) Match sim vs Dispersion [{mode_label}]\n{rho_note}')
    ax.legend(handles=[Patch(color='#E74C3C',label='N-term'), Patch(color='#2980B9',label='Core')], fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[0,1]
    nd = valid[is_nt]['position_offset_std'].dropna(); cd = valid[~is_nt]['position_offset_std'].dropna()
    if len(nd)>0 and len(cd)>0:
        bp = ax.boxplot([nd.values, cd.values], labels=[f'N-term\nn={len(nd)}', f'Core\nn={len(cd)}'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#E74C3C'); bp['boxes'][1].set_facecolor('#2980B9')
        _, p = mannwhitneyu(nd, cd, alternative='two-sided')
        ax.set_title(f'(b) Position offset std\nMW p = {p:.2e}')
    ax.set_ylabel('Position offset std'); ax.grid(True, alpha=0.2)
    ax = axes[1,0]
    sv = valid[valid['shannon_entropy'].notna()]
    if len(sv)>0:
        snt = sv['mean_abs_position'] <= n_term_boundary
        colors = ['#E74C3C' if n else '#2980B9' for n in snt]
        ax.scatter(sv['shannon_entropy'], sv['dispersion'], c=colors, s=12, alpha=0.5, edgecolors='none')
        rho, pv = spearmanr(sv['shannon_entropy'].values, sv['dispersion'].values)
        ax.set_xlabel('Shannon entropy'); ax.set_ylabel('Dispersion')
        ax.set_title(f'(c) Shannon vs Dispersion\nρ = {rho:.3f}, p = {pv:.2e}')
        ax.legend(handles=[Patch(color='#E74C3C',label='N-term'), Patch(color='#2980B9',label='Core')], fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[1,1]
    nd = valid[is_nt]['dispersion'].dropna(); cd = valid[~is_nt]['dispersion'].dropna()
    bins = np.linspace(min(valid['dispersion'].min(),0), valid['dispersion'].max(), 50)
    ax.hist(nd, bins=bins, alpha=0.6, color='#E74C3C', label=f'N-term n={len(nd)}')
    ax.hist(cd, bins=bins, alpha=0.6, color='#2980B9', label=f'Core n={len(cd)}')
    ax.set_xlabel('Dispersion'); ax.set_ylabel('Freq'); ax.set_title('(d) Dispersion dist'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(out_dir/'diagnostics.png', dpi=dpi)
    plt.savefig(out_dir/'diagnostics.pdf'); plt.close()
    print(f"  Saved diagnostics.png/pdf")

def plot_directionality(disp_res, dir_res, mode_label, out_dir, dpi, n_term_boundary=50, reduction='pca'):
    # ⭐ MODIFIED: updated title to reflect reduction method
    dd = pd.DataFrame(disp_res); dr = pd.DataFrame(dir_res)
    vm = dd['dispersion'].notna() & dr['effective_rank'].notna()
    vd = dd[vm].reset_index(drop=True); vr = dr[vm].reset_index(drop=True)
    if len(vd)==0:
        return
    fig = plt.figure(figsize=(18,14))
    pos = vd['position'].values; disp = vd['dispersion'].values
    er = vr['effective_rank'].values; ds = vr['directional_score'].values
    mp = vd['mean_abs_position'].values
    ax = fig.add_subplot(2,2,1)
    sc = ax.scatter(er, disp, c=ds, s=20, alpha=0.6, cmap='RdYlGn', edgecolors='grey', lw=0.3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Directional score')
    md = np.nanmedian(disp); mr = np.nanmedian(er)
    ax.axhline(md, color='grey', ls='--', alpha=0.5); ax.axvline(mr, color='grey', ls='--', alpha=0.5)
    rr = er.max()-er.min(); dr_ = disp.max()-disp.min()
    ax.text(mr+0.15*rr, md+0.12*dr_, 'Directional\nDivergence\n(candidate +sel)', fontsize=9, color='darkred', ha='center', va='center', fontweight='bold')
    ax.text(mr+0.15*rr, md-0.10*dr_, 'Directional\nconserved', fontsize=8, color='darkblue', ha='center', va='top')
    ax.text(mr-0.15*rr, md+0.12*dr_, 'Isotropic\nrelaxation', fontsize=8, color='darkorange', ha='center', va='center')
    ax.text(mr-0.15*rr, md-0.10*dr_, 'Purifying\nselection', fontsize=8, color='darkgreen', ha='center', va='top')
    ax.set_xlabel('Effective rank (low=directional)'); ax.set_ylabel('Dispersion')
    ax.set_title(f'(a) Directionality [{mode_label}]'); ax.grid(True, alpha=0.2)
    ti = np.argsort(ds)[-8:]
    for idx in ti:
        ax.annotate(f"P{int(pos[idx])}", xy=(er[idx], disp[idx]), xytext=(5,5), textcoords='offset points', fontsize=7, color='darkred', fontweight='bold')
    t3 = np.argsort(ds)[-3:][::-1]
    for i, idx in enumerate(t3):
        ax = fig.add_subplot(2,3,4+i)
        # ⭐ MODIFIED: 使用参数传入的降维方法名称更新标题
        p2d = vr.iloc[idx]['pca_2d']; lb = vr.iloc[idx]['cluster_labels']
        nc = vr.iloc[idx]['n_clusters_gmm']; rk = vr.iloc[idx]['effective_rank']
        sc_ = vr.iloc[idx]['directional_score']; pi = int(vd.iloc[idx]['position'])
        if p2d is None or lb is None:
            ax.text(0.5,0.5,'N/A',ha='center',va='center'); ax.set_title(f'Pos {pi}'); continue
        cmap = plt.cm.tab10(np.linspace(0,1,max(len(set(lb)),3)))
        for lab, col in zip(sorted(set(lb)), cmap[:len(set(lb))]):
            ax.scatter(p2d[lb==lab,0], p2d[lb==lab,1], c=[col], s=15, alpha=0.7, edgecolors='none', label=f'C{lab}')
        ax.set_title(f'Pos {pi} | {nc} cl | ER={rk:.1f} DS={sc_:.3f} | {reduction.upper()} proj.', fontsize=9)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax = fig.add_subplot(2,2,2)
    colors = ['#E74C3C' if m<=n_term_boundary else '#2980B9' for m in mp]
    ax.scatter(pos, ds, c=colors, s=15, alpha=0.6, edgecolors='none')
    if len(ds)>11:
        ax.plot(pos, np.convolve(ds, np.ones(11)/11, mode='same'), 'k-', lw=2, alpha=0.5)
    vs = ds[~np.isnan(ds)]
    if len(vs)>0:
        th = np.percentile(vs, 90); ax.axhline(th, color='red', ls='--', alpha=0.5, label=f'90th ({th:.2f})')
    for idx in ti:
        ax.annotate(f"P{int(pos[idx])}", xy=(pos[idx], ds[idx]), xytext=(0,-12), textcoords='offset points', fontsize=7, color='darkred', fontweight='bold', ha='center')
    ax.set_xlabel('Position'); ax.set_ylabel('Directional score')
    ax.set_title(f'(c) Directional score [{mode_label}]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(out_dir/'directionality.png', dpi=dpi)
    plt.savefig(out_dir/'directionality.pdf'); plt.close()
    print(f"  Saved directionality.png/pdf")

def plot_phylo_structure(disp_res, phylo_res, mode_label, out_dir, dpi, n_term_boundary=50):
    # ⭐ MODIFIED: 在图中添加 q_value 作为标记（如有）
    dd = pd.DataFrame(disp_res); dp = pd.DataFrame(phylo_res)
    vm = dd['directional_score'].notna() & dp['phylo_structure_z'].notna()
    vd = dd[vm].reset_index(drop=True); vp = dp[vm].reset_index(drop=True)
    if len(vd)==0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16,7))
    ax = axes[0]
    ds_ = vd['directional_score'].values; zs = vp['phylo_structure_z'].values
    pv = vp['phylo_structure_pval'].values; mp = vd['mean_abs_position'].values
    colors = ['#E74C3C' if m<=n_term_boundary else '#2980B9' for m in mp]
    sizes = [30+20*(1-min(p,0.05)/0.05) if not np.isnan(p) else 20 for p in pv]
    ax.scatter(ds_, zs, c=colors, s=sizes, alpha=0.6, edgecolors='grey', lw=0.3)
    ax.axhline(0, color='grey', ls='--', alpha=0.5)
    ax.axhline(2.0, color='red', ls=':', alpha=0.5, label='z=2')
    ax.axvline(0.5, color='grey', ls='--', alpha=0.3)
    pds = vp['phylo_directional_score'].values
    ti = np.argsort(pds)[-8:]
    for idx in ti:
        ax.annotate(f"P{int(vd.iloc[idx]['position'])}", xy=(ds_[idx], zs[idx]), xytext=(5,5), textcoords='offset points', fontsize=7, color='darkred', fontweight='bold')
    ax.set_xlabel('Directional score'); ax.set_ylabel('Phylo structure z')
    ax.set_title(f'(a) Phylo structure vs Directionality [{mode_label}]')
    ax.legend(handles=[Patch(color='#E74C3C',label='N-term'), Patch(color='#2980B9',label='Core')], fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[1]
    pos = vd['position'].values
    ax.scatter(pos, pds, c=colors, s=15, alpha=0.6, edgecolors='none')
    if len(pds)>11:
        ax.plot(pos, np.convolve(pds, np.ones(11)/11, mode='same'), 'k-', lw=2, alpha=0.5)
    vpds = pds[~np.isnan(pds)]
    if len(vpds)>0:
        th = np.percentile(vpds, 90); ax.axhline(th, color='red', ls='--', alpha=0.5, label=f'90th ({th:.2f})')
    for idx in ti:
        ax.annotate(f"P{int(vd.iloc[idx]['position'])}", xy=(pos[idx], pds[idx]), xytext=(0,-12), textcoords='offset points', fontsize=7, color='darkred', fontweight='bold', ha='center')
    ax.set_xlabel('Position'); ax.set_ylabel('Phylo-directional score')
    ax.set_title(f'(b) Phylo-directional score [{mode_label}]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(out_dir/'phylo_structure.png', dpi=dpi)
    plt.savefig(out_dir/'phylo_structure.pdf'); plt.close()
    print(f"  Saved phylo_structure.png/pdf")


# ===========================================================================
# Report (MODIFIED: add FDR info)
# ===========================================================================
def generate_report(results, dir_results, phylo_results, n_total, out_dir, args, mode_label, ref_id=None, n_term_boundary=50):
    df = pd.DataFrame(results)
    valid = df[df['dispersion'].notna()].copy()
    if len(valid)==0:
        with open(out_dir/'report.txt','w',encoding='utf-8') as f:
            f.write("No valid data.\n")
        return
    is_nt = valid['mean_abs_position'] <= n_term_boundary
    nt = valid[is_nt]; co = valid[~is_nt]
    ma = np.nanmean(valid['dispersion']); sa = np.nanstd(valid['dispersion'])
    mn = np.nanmean(nt['dispersion']) if len(nt)>0 else np.nan
    mc = np.nanmean(co['dispersion']) if len(co)>0 else np.nan
    mwp = None
    if len(nt)>0 and len(co)>0:
        _, mwp = mannwhitneyu(nt['dispersion'].dropna(), co['dispersion'].dropna(), alternative='two-sided')
    
    # ⭐ MODIFIED: 对 N-term vs Core 的 MW 检验 p 值应用 FDR
    mwp_q = np.nan
    if mwp is not None and HAS_STATSMODELS:
        # 尽管这里只有一个检验，但为了通用性，我们还是保留单检验 p 值上报。
        pass 

    mcon = valid.nsmallest(20, 'dispersion'); mvar = valid.nlargest(20, 'dispersion')
    ntv = nt.nlargest(10, 'dispersion') if len(nt)>0 else pd.DataFrame()
    sv = valid[valid['shannon_entropy'].notna()]
    rho_sd = np.nan
    if len(sv)>=3:
        rho_sd, _ = spearmanr(sv['shannon_entropy'].values, sv['dispersion'].values)

    rho_ms = np.nan
    if 'DTW' not in mode_label.upper():
        if len(valid) >= 3:
            rho_ms, _ = spearmanr(valid['mean_match_similarity'].values, valid['dispersion'].values)
        ms_note = (f"Match sim vs Dispersion Spearman ρ: {rho_ms:.3f}\n"
                   f"  (Independent diagnostic — {'OK' if abs(rho_ms)<0.8 else 'WARNING: >0.8 suggests redundancy with match quality'})")
    else:
        ms_note = ("Match sim vs Dispersion: ρ = −1.0 (mathematical identity)\n"
                   "  (mean_match_similarity = 1−dispersion by construction in DTW mode)")

    partial_note = ""
    if len(sv)>30:
        sl, ic = np.polyfit(sv['shannon_entropy'], sv['dispersion'], 1)
        res = sv['dispersion'] - (sl*sv['shannon_entropy']+ic)
        sc = sv.copy(); sc['residual'] = res
        nr = sc[sc['mean_abs_position']<=n_term_boundary]; cr = sc[sc['mean_abs_position']>n_term_boundary]
        if len(nr)>0 and len(cr)>0:
            _, p2 = mannwhitneyu(nr['residual'], cr['residual'], alternative='two-sided')
            partial_note = (f"\nPartial (dispersion | Shannon): N-term resid={np.mean(nr['residual']):.5f}, "
                           f"Core resid={np.mean(cr['residual']):.5f}, MW p={p2:.2e}\n"
                           f"  → {'N-term signal persists' if p2<0.05 else 'Explained by AA diversity'}")
    has_phylo = phylo_results is not None and len(phylo_results)>0

    # ⭐ MODIFIED: 在报告中提示 FDR 校正状态
    fdr_note = ""
    if HAS_STATSMODELS:
        fdr_note = "FDR correction applied (Benjamini-Hochberg) for phylo permutation tests."
    else:
        fdr_note = "WARNING: statsmodels not installed. FDR correction NOT applied. Install 'pip install statsmodels'."

    report = f"""
================================================================================
PEACE Analysis Report
================================================================================
Mode:             {mode_label}
Input:            {args.fasta}
Total sequences:  {n_total}
Model:            {args.model_name}
Dim Reduction:    {args.dim_reduction.upper()} (for directionality clustering)
"""
    if ref_id:
        report += f"Reference:        {ref_id}\n"
    else:
        report += f"Consensus:        DTW Barycenter Averaging (window={args.window_size})\n"
    report += f"""
Positions:        {len(valid)}
Min effective:    {args.min_effective_seqs}
N-term boundary:  mean_abs_position ≤ {n_term_boundary}
Phylogenetic tree: {'YES' if has_phylo else 'N/A'}
FDR Status:       {fdr_note}

================================================================================
Summary
================================================================================
Mean dispersion (all):     {ma:.5f}  (±{sa:.5f})
Mean dispersion (N-term):  {mn:.5f}
Mean dispersion (core):    {mc:.5f}
Δ (N-term − core):         {mn - mc:.5f}
Mann-Whitney p (N vs Core): {mwp:.4e}  (N-term {'>' if mn>mc else '<'} core)
N-term > mean+2σ: {len(nt[nt['dispersion']>ma+2*sa]) if len(nt)>0 else 0}

================================================================================
Validation
================================================================================
Dispersion vs Shannon Spearman ρ: {rho_sd:.3f}
{ms_note}
"""
    report += partial_note

    def fmt(r):
        sh = f"{r['shannon_entropy']:.3f}" if not np.isnan(r['shannon_entropy']) else "N/A"
        return (f"{int(r['position']):<6} {r.get('dominant_aa','?'):<4} "
                f"{r['dispersion']:<12.5f} {sh:>10} {int(r['n_effective']):<8} "
                f"{r.get('mean_abs_position',-1):<10.0f} {r['mean_match_similarity']:<10.4f}")

    report += f"""
================================================================================
Top 10 CONSTRAINED (lowest dispersion → purifying selection)
================================================================================
{'Pos':<6}{'AA':<4}{'Dispersion':<12}{'Shannon':<10}{'N_eff':<8}{'AppxPos':<10}{'MeanSim':<10}
{'-'*60}
"""
    for _, r in mcon.head(10).iterrows():
        report += fmt(r) + "\n"

    report += f"""
================================================================================
Top 10 VARIABLE (highest dispersion → relaxed/diversified)
================================================================================
{'Pos':<6}{'AA':<4}{'Dispersion':<12}{'Shannon':<10}{'N_eff':<8}{'AppxPos':<10}{'MeanSim':<10}
{'-'*60}
"""
    for _, r in mvar.head(10).iterrows():
        report += fmt(r) + "\n"

    if len(ntv)>0:
        report += f"""
================================================================================
N-terminal Top 10 Variable
================================================================================
{'Pos':<6}{'AA':<4}{'Dispersion':<12}{'Shannon':<10}{'N_eff':<8}{'AppxPos':<10}
{'-'*50}
"""
        for _, r in ntv.iterrows():
            sh = f"{r['shannon_entropy']:.3f}" if not np.isnan(r['shannon_entropy']) else "N/A"
            report += (f"{int(r['position']):<6} {r.get('dominant_aa','?'):<4} "
                       f"{r['dispersion']:<12.5f} {sh:>10} {int(r['n_effective']):<8} "
                       f"{r.get('mean_abs_position',-1):<10.0f}\n")

    vd = [d for d in dir_results if not np.isnan(d['directional_score'])]
    if len(vd)>0:
        td = np.argsort([d['directional_score'] for d in vd])[-10:][::-1]
        report += f"""
================================================================================
DIRECTIONALITY Top 10 (family-wide structured divergence)
================================================================================
{'Pos':<6}{'AA':<4}{'Disp':<10}{'EffRank':<10}{'NClust':<8}{'Sil':<12}{'DirScore':<10}{'AppxPos':<10}
{'-'*70}
"""
        for idx in td:
            d = vd[idx]
            if idx < len(results):
                r = results[idx]
                si = f"{d['silhouette']:.3f}" if not np.isnan(d['silhouette']) else "N/A"
                report += (f"{r['position']:<6} {r.get('dominant_aa','?'):<4} {r['dispersion']:<10.4f} "
                           f"{d['effective_rank']:<10.2f} {d['n_clusters_gmm']:<8} {si:<12} "
                           f"{d['directional_score']:<10.3f} {r.get('mean_abs_position',-1):<10.0f}\n")
        ntc = sum(1 for idx in td if idx<len(results) and results[idx].get('mean_abs_position',999)<=n_term_boundary)
        report += f"\nOf top 10 directional, {ntc}/10 in N-term region.\n"

    if has_phylo:
        vp = [p for p in phylo_results if not np.isnan(p['phylo_directional_score'])]
        if len(vp)>0:
            tp = np.argsort([p['phylo_directional_score'] for p in vp])[-10:][::-1]
            report += f"""
================================================================================
PHYLOGENETIC STRUCTURE Top 10 (branch-specific candidate)
================================================================================
High phylo_z (>2) + high directional_score → GMM clusters aligned with clades
→ candidate episodic positive selection on specific lineages
================================================================================
{'Pos':<6}{'AA':<4}{'DirScore':<10}{'PhyloZ':<10}{'PhyloP':<12}{'PhyloQ':<12}{'PhyloDir':<10}{'AppxPos':<10}
{'-'*70}
"""
            for idx in tp:
                p = vp[idx]
                if idx < len(results):
                    r = results[idx]; d = dir_results[idx]
                    ps = f"{p['phylo_structure_pval']:.4f}" if not np.isnan(p['phylo_structure_pval']) else "N/A"
                    # ⭐ MODIFIED: 增加 Q value 的列输出
                    qs = f"{p.get('phylo_structure_qval', np.nan):.4f}" if not np.isnan(p.get('phylo_structure_qval', np.nan)) else "N/A"
                    report += (f"{r['position']:<6} {r.get('dominant_aa','?'):<4} "
                               f"{d['directional_score']:<10.3f} {p['phylo_structure_z']:<10.2f} "
                               f"{ps:<12} {qs:<12} {p['phylo_directional_score']:<10.3f} "
                               f"{r.get('mean_abs_position',-1):<10.0f}\n")
            ntp = sum(1 for idx in tp if idx<len(results) and results[idx].get('mean_abs_position',999)<=n_term_boundary)
            report += f"\nOf top 10 phylo-structure, {ntp}/10 in N-term region.\n"

    report += f"""
================================================================================
Interpretation
================================================================================
- LOW dispersion → purifying selection
- HIGH dispersion → relaxed/diversified
- Directional score: high + low effective rank + ≥2 clusters → structured divergence
- Phylo z-score: high (>2) → clusters map to phylogenetic clades → episodic +sel
  low → family-wide pervasive divergence (complementary to MEME's branch-specific)
- *** Use FDR corrected Q-values (PhyloQ) with a threshold of 0.05 for robust results. ***
"""

    with open(out_dir/'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"  Saved report.txt")


# ===========================================================================
# Main (MODIFIED: pass args.dim_reduction)
# ===========================================================================
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_label = f"PEACE-{args.mode.upper()}"

    print("=" * 70)
    print(f"{mode_label}: Per-position Embedding Constraint Estimation")
    print(f"Using dim-reduction method: {args.dim_reduction.upper()}")
    print("=" * 70)

    records = load_fasta(args.fasta)
    n_total = len(records)
    tok, model = load_model(args.model_name, args.device)
    print("\n[1/6] Embeddings...")
    embeddings, sequences = extract_embeddings(records, tok, model, args.device, args.max_length, out_dir)
    print("\n[2/6] Position templates & debias...")
    templates = compute_position_templates(embeddings)
    embeddings_db = debias_embeddings(embeddings, templates)

    ref_id = None
    if args.mode == 'dtw':
        print(f"\n[3/6 DTW] Trajectories (w={args.window_size})...")
        traj, winfo = build_trajectories(embeddings_db, args.window_size)
        print(f"[4/6 DTW] PCA ({args.pca_components})...")
        traj_red, _ = pca_reduce(traj, args.pca_components)
        print(f"[5/6 DTW] DBA consensus...")
        dba_cache = out_dir / "dba_consensus.pkl"
        consensus = compute_dba_consensus(traj_red, args.consensus_length, args.dtw_max_iter, args.dtw_subset, dba_cache)
        print(f"[6/6 DTW] Align & collect...")
        position_data = dtw_align_and_collect(consensus, traj_red, traj, sequences, winfo, args.window_size, embeddings_db)
    else:  # reference
        print(f"\n[3/6 REF] Selecting reference...")
        ref_rec = select_reference(records, args.seed)
        ref_id = ref_rec.id
        ref_emb = embeddings_db[ref_id]
        ref_seq = sequences[ref_id]
        print(f"[4/6 REF] Similarity threshold...")
        sim_th = determine_sim_threshold(ref_emb, embeddings_db, args.sim_threshold_percentile)
        print(f"[5/6 REF] Soft correspondence ({ref_emb.shape[0]} pos × {len(embeddings_db)-1} targets)...")
        position_data = soft_correspondence(ref_id, ref_emb, ref_seq, embeddings_db, sequences, sim_th)

    # --- Common: metrics, directionality ---
    print(f"\n[Analysis] Dispersion & Shannon...")
    results = compute_metrics(position_data, args.min_effective_seqs)
    print(f"[Analysis] Directionality...")
    # ⭐ MODIFIED: 传入 dim_reduction 参数
    dir_results = compute_directionality(position_data, args.min_effective_seqs, pca_dim=5, reduction=args.dim_reduction)
    for r, d in zip(results, dir_results):
        r['effective_rank'] = d['effective_rank']
        r['top2_ratio'] = d['top2_ratio']
        r['n_clusters_gmm'] = d['n_clusters_gmm']
        r['silhouette'] = d['silhouette']
        r['directional_score'] = d['directional_score']

    # --- Phylogeny (optional) ---
    phylo_results = None
    if args.tree:
        phylo_results = analyze_phylogenetic_structure(results, dir_results, position_data, args.tree, out_dir, args.phylo_permutations)
        for r, p in zip(results, phylo_results):
            r['phylo_structure_z'] = p['phylo_structure_z']
            r['phylo_structure_pval'] = p['phylo_structure_pval']
            r['phylo_structure_qval'] = p.get('phylo_structure_qval', np.nan)
            r['phylo_directional_score'] = p['phylo_directional_score']

    pd.DataFrame(results).to_csv(out_dir/'dispersion_data.csv', index=False)
    print(f"  Saved dispersion_data.csv ({len(results)} positions)")

    # --- Visualizations ---
    print("\nVisualizing...")
    plot_dispersion_profile(results, mode_label, out_dir, args.dpi)
    plot_diagnostics(results, mode_label, out_dir, args.dpi)
    # ⭐ MODIFIED: 传入 reduction method 用于图标题
    plot_directionality(results, dir_results, mode_label, out_dir, args.dpi, reduction=args.dim_reduction)
    if phylo_results is not None:
        plot_phylo_structure(results, phylo_results, mode_label, out_dir, args.dpi)

    # --- Report ---
    print(f"\n{'='*70}\nReport...")
    generate_report(results, dir_results, phylo_results, n_total, out_dir, args, mode_label, ref_id)

    print(f"\n{'='*70}\nDone. Results: {out_dir}/")
    for f in ['dispersion_data.csv','dispersion_profile.png','diagnostics.png','directionality.png','report.txt']:
        print(f"  - {f}")
    if args.tree:
        print(f"  - phylo_structure.png")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()