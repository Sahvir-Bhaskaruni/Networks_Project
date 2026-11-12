#!/usr/bin/env python3
"""
Mapping Gene-Regulatory Network Influence on Neural Network Connectivity Code.

Usage
-----
  python grn_connectome_analysis.py demo --out results_demo
  python grn_connectome_analysis.py run \
      --connectome edges.csv --meta neurons.csv --expression counts.csv \
      --regulons scenic=scenic_regulons.gmt --regulons wormbase=wb_regulatory.csv \
      --out results

Input formats (CSV unless noted)
  edges.csv     pre,post[,weight]          directed chemical/gap-junction edges between neurons
  neurons.csv   neuron,type[,class]        type in {sensory, interneuron, motor, pharyngeal};
                                           'class' optional (e.g. ADAL/ADAR -> ADA)
  counts.csv    genes x columns raw counts. Columns are neurons, or neuron classes if
                neurons.csv has a 'class' column (profiles are then copied to each neuron).
                For single-cell data, first collapse with aggregate_cells().
  regulons      .gmt (name<TAB>desc<TAB>genes...) or CSV with columns tf,target
                (gene set = TF + targets). Any number of --regulons name=path.

Interpretation choices where the paper leaves room (all easy to change below)
  * Expression-edge weight between neurons i,j:
        w_ij = sum over genes g in TopX(i) & TopX(j) of (log2FC_gi + log2FC_gj)/2
    where TopX(i) = the X genes with highest one-vs-all log2FC in neuron i.
  * The dense weighted network is thresholded to the same number of edges as the
    connectome before assortativity is computed, so networks are density-matched.
  * Connectome is analysed as an undirected projection for assortativity.
  * Node-specific assortativity = Piraveenan local degree assortativity (sums to global r)
    plus the fraction of a node's neighbours sharing its neuron type.
  * Significance = degree-preserving edge-swap null.
  * GSEA is a weighted KS running-sum with gene-set permutation (no external dependency).
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

TYPES = ["sensory", "interneuron", "motor", "pharyngeal"]


# ----------------------------------------------------------------------------
# Expression utilities
# ----------------------------------------------------------------------------
def aggregate_cells(counts, cell_to_neuron):
    """genes x cells counts + Series(cell -> neuron/class) -> genes x neurons summed counts."""
    m = cell_to_neuron.reindex(counts.columns)
    return counts.loc[:, m.notna().values].T.groupby(m.dropna().values).sum().T


def cpm(counts):
    return counts / counts.sum(axis=0) * 1e6


def one_vs_all_log2fc(counts, pseudo=1.0):
    """log2 fold change of each column (CPM) vs the mean of all other columns."""
    c = cpm(counts)
    n = c.shape[1]
    rest = (c.sum(axis=1).values[:, None] - c.values) / (n - 1)
    return pd.DataFrame(np.log2((c.values + pseudo) / (rest + pseudo)),
                        index=c.index, columns=c.columns)


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
def build_connectome(edges, meta_types):
    G = nx.Graph()
    G.add_nodes_from(meta_types.index)
    for r in edges.itertuples(index=False):
        a, b = r[0], r[1]
        w = r[2] if len(r) > 2 else 1.0
        if a in G and b in G and a != b:
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)
    nx.set_node_attributes(G, meta_types.to_dict(), "type")
    return G


def expression_weights(lfc, X):
    """Dense symmetric weight matrix (nodes x nodes) from top-X log2FC genes."""
    M = lfc.values
    X = min(X, M.shape[0])
    top = np.zeros(M.shape, dtype=bool)
    np.put_along_axis(top, np.argsort(-M, axis=0)[:X], True, axis=0)
    L = np.where(top, M, 0.0)
    T = top.astype(float)
    W = (L.T @ T + T.T @ L) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


def threshold_graph(W, nodes, types, n_edges):
    iu = np.triu_indices_from(W, k=1)
    w = W[iu]
    order = np.argsort(-w)[:n_edges]
    order = order[w[order] > 0]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for k in order:
        G.add_edge(nodes[iu[0][k]], nodes[iu[1][k]], weight=float(w[k]))
    nx.set_node_attributes(G, types.to_dict(), "type")
    return G


# ----------------------------------------------------------------------------
# Assortativity
# ----------------------------------------------------------------------------
def _safe(f, *a, **k):
    try:
        v = f(*a, **k)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def local_assortativity(G):
    """Piraveenan-style local degree assortativity; values sum to global r."""
    deg = dict(G.degree())
    M = G.number_of_edges()
    if M == 0:
        return pd.Series(dtype=float)
    rem = {v: deg[v] - 1 for v in G}
    two_m = 2 * M
    mu = sum(rem[v] * deg[v] for v in G) / two_m
    var = sum(rem[v] ** 2 * deg[v] for v in G) / two_m - mu ** 2
    out = {}
    for v in G:
        if deg[v] == 0 or var == 0:
            out[v] = np.nan
            continue
        out[v] = rem[v] * sum(rem[u] - mu for u in G[v]) / (two_m * var)
    return pd.Series(out)


def neighbour_type_homophily(G):
    out = {}
    for v in G:
        nb = list(G[v])
        out[v] = np.mean([G.nodes[u]["type"] == G.nodes[v]["type"] for u in nb]) if nb else np.nan
    return pd.Series(out)


def assort_stats(G, n_perm, seed):
    obs_t = _safe(nx.attribute_assortativity_coefficient, G, "type")
    obs_d = _safe(nx.degree_assortativity_coefficient, G)
    M = G.number_of_edges()
    rng = random.Random(seed)
    nt, nd = [], []
    if M >= 4 and n_perm > 0:
        for _ in range(n_perm):
            H = G.copy()
            try:
                nx.double_edge_swap(H, nswap=5 * M, max_tries=100 * M, seed=rng.randrange(1 << 30))
            except Exception:
                continue
            nt.append(_safe(nx.attribute_assortativity_coefficient, H, "type"))
            nd.append(_safe(nx.degree_assortativity_coefficient, H))

    def pval(obs, null):
        null = np.array([x for x in null if np.isfinite(x)])
        if not np.isfinite(obs) or len(null) == 0:
            return np.nan
        return (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (1 + len(null))

    return dict(type_assort=obs_t, type_assort_p=pval(obs_t, nt),
                degree_assort=obs_d, degree_assort_p=pval(obs_d, nd))


# ----------------------------------------------------------------------------
# Analysis 1  (+ network cache reused by Analysis 3)
# ----------------------------------------------------------------------------
def analysis1(conn, lfc, types, sources, Xs, n_perm, out, seed):
    nodes = list(conn.nodes)
    n_edges = conn.number_of_edges()
    rows, nets = [], {}

    base = assort_stats(conn, n_perm, seed)
    rows.append(dict(source="connectome", X=np.nan, n_edges=n_edges, **base))
    pd.DataFrame({"local_degree_assort": local_assortativity(conn),
                  "neighbour_type_homophily": neighbour_type_homophily(conn)}
                 ).to_csv(out / "node_assortativity_connectome.csv")

    for sname, genes in sources.items():
        sub = lfc if genes is None else lfc.loc[lfc.index.intersection(list(genes))]
        if sub.shape[0] < 5:
            print(f"  skipping source {sname}: <5 genes overlap")
            continue
        for X in Xs:
            W = expression_weights(sub, X)
            G = threshold_graph(W, nodes, types, n_edges)
            nets[(sname, X)] = (G, W)
            st = assort_stats(G, n_perm, seed)
            rows.append(dict(source=sname, X=X, n_edges=G.number_of_edges(), **st))
            if X == Xs[len(Xs) // 2]:
                pd.DataFrame({"local_degree_assort": local_assortativity(G),
                              "neighbour_type_homophily": neighbour_type_homophily(G)}
                             ).to_csv(out / f"node_assortativity_{sname}_X{X}.csv")
        print(f"  analysis 1: {sname} done")

    res = pd.DataFrame(rows)
    res.to_csv(out / "analysis1_assortativity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["type_assort", "degree_assort"],
                              ["Neuron-type assortativity", "Degree assortativity"]):
        for sname in sources:
            d = res[res.source == sname].sort_values("X")
            ax.plot(d.X, d[col], marker="o", label=sname)
        ax.axhline(res.loc[res.source == "connectome", col].iloc[0], color="k", ls="--",
                   label="connectome")
        ax.set_xscale("log")
        ax.set_xlabel("X (top DE genes per neuron)")
        ax.set_ylabel(col)
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "analysis1_assortativity_vs_X.png", dpi=150)
    plt.close(fig)
    return res, nets


# ----------------------------------------------------------------------------
# Analysis 2: GSEA by neuron type, then gene-level comparison
# ----------------------------------------------------------------------------
def _running_es(mask, w):
    hw = np.where(mask, w, 0.0)
    s = hw.sum()
    n_miss = (~mask).sum()
    if s == 0 or n_miss == 0:
        return 0.0
    dev = np.cumsum(hw) / s - np.cumsum(~mask) / n_miss
    return float(dev[np.argmax(np.abs(dev))])


def _bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def gsea(metric, sets, n_perm, rng, min_size=5):
    r = metric.sort_values(ascending=False)
    genes, w = r.index.values, np.abs(r.values)
    N = len(genes)
    cache, rows = {}, []
    for name, gs in sets.items():
        mask = np.isin(genes, list(gs))
        k = int(mask.sum())
        if k < min_size or k >= N:
            continue
        es = _running_es(mask, w)
        if k not in cache:
            null = []
            for _ in range(n_perm):
                m = np.zeros(N, bool)
                m[rng.choice(N, k, replace=False)] = True
                null.append(_running_es(m, w))
            cache[k] = np.array(null)
        null = cache[k]
        side = null[null > 0] if es >= 0 else -null[null < 0]
        nes = es / side.mean() if len(side) and side.mean() > 0 else np.nan
        p = (1 + np.sum(np.abs(null) >= abs(es))) / (1 + n_perm)
        rows.append(dict(gene_set=name, size=k, ES=es, NES=nes, p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["fdr"] = _bh(df.p.values)
    return df


def analysis2(counts, types, lfc, reg_sets, n_perm, out, seed, fdr=0.05, topn=5):
    rng = np.random.default_rng(seed)
    agg = counts.T.groupby(types.reindex(counts.columns).values).sum().T
    tlfc = one_vs_all_log2fc(agg)                      # genes x types (CPM aggregate)
    all_enr, summ = [], []
    for t in tlfc.columns:
        g = gsea(tlfc[t], reg_sets, n_perm, rng)
        if not len(g):
            continue
        g.insert(0, "type", t)
        all_enr.append(g)
        sig = g[g.fdr < fdr]
        if sig.empty:
            continue
        universe = sorted(set().union(*[reg_sets[s] for s in sig.gene_set]) & set(tlfc.index))
        members = types.index[types == t]
        t_vec = tlfc.loc[universe, t]
        t_top = set(t_vec.abs().nlargest(topn).index)
        null_abs = lfc.loc[universe].abs().values.mean()   # average |log2FC| across all neurons
        for nrn in lfc.columns:
            n_vec = lfc.loc[universe, nrn]
            n_top = set(n_vec.abs().nlargest(topn).index)
            rho = stats.spearmanr(n_vec, t_vec)[0] if len(universe) > 2 else np.nan
            summ.append(dict(type=t, neuron=nrn, is_member=nrn in members,
                             n_sig_sets=len(sig), n_genes=len(universe),
                             spearman_vs_type=rho,
                             top5_overlap_jaccard=len(n_top & t_top) / len(n_top | t_top),
                             neuron_top5_mean_abs=n_vec.abs().nlargest(topn).mean(),
                             type_top5_mean_abs=t_vec.abs().nlargest(topn).mean(),
                             null_mean_abs=null_abs))
    enr = pd.concat(all_enr) if all_enr else pd.DataFrame()
    enr.to_csv(out / "analysis2_gsea_by_type.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "analysis2_neuron_vs_type_enrichment.csv", index=False)
    if len(sm):
        agg_tab = sm.groupby(["type", "is_member"])[["spearman_vs_type", "top5_overlap_jaccard"]].mean()
        agg_tab.to_csv(out / "analysis2_member_vs_nonmember_summary.csv")
        print(agg_tab.round(3).to_string())
    return enr, sm


# ----------------------------------------------------------------------------
# Analysis 3: community structure vs canonical neuron types
# ----------------------------------------------------------------------------
def groupmate_score(null_labels, comm_labels):
    """Per-neuron fraction of its null-model group-mates that share its community."""
    nl, cl = np.asarray(null_labels), np.asarray(comm_labels)
    sc = np.full(len(nl), np.nan)
    for i in range(len(nl)):
        mates = np.where((nl == nl[i]) & (np.arange(len(nl)) != i))[0]
        if len(mates):
            sc[i] = np.mean(cl[mates] == cl[i])
    return sc


def community_eval(null_labels, comm, n_perm, rng):
    sc = groupmate_score(null_labels, comm)
    obs = np.nanmean(sc)
    null = [np.nanmean(groupmate_score(null_labels, rng.permutation(comm))) for _ in range(n_perm)]
    return sc, dict(score=obs, null_mean=np.mean(null),
                    p=(1 + np.sum(np.array(null) >= obs)) / (1 + n_perm),
                    ARI=adjusted_rand_score(null_labels, comm),
                    NMI=normalized_mutual_info_score(null_labels, comm))


def analysis3(conn, nets, types, k, n_perm, out, seed):
    rng = np.random.default_rng(seed)
    nodes = list(conn.nodes)
    null_labels = types.reindex(nodes).values
    rows = []

    comms = nx.community.greedy_modularity_communities(conn, weight="weight", cutoff=k, best_n=k)
    lab = {n: i for i, c in enumerate(comms) for n in c}
    _, st = community_eval(null_labels, np.array([lab[n] for n in nodes]), n_perm, rng)
    rows.append(dict(source="connectome", X=np.nan, **st))

    per_neuron = {}
    for (sname, X), (G, W) in nets.items():
        A = W + 1e-6
        np.fill_diagonal(A, 0.0)
        comm = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                                  assign_labels="kmeans").fit_predict(A)
        sc, st = community_eval(null_labels, comm, n_perm, rng)
        rows.append(dict(source=sname, X=X, **st))
        per_neuron[f"{sname}_X{X}"] = sc
    res = pd.DataFrame(rows)
    res.to_csv(out / "analysis3_community_scores.csv", index=False)
    pd.DataFrame(per_neuron, index=nodes).to_csv(out / "analysis3_per_neuron_scores.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    for sname in res.source.unique():
        if sname == "connectome":
            continue
        d = res[res.source == sname].sort_values("X")
        ax.plot(d.X, d.score, marker="o", label=sname)
    ax.axhline(res.loc[res.source == "connectome", "score"].iloc[0], color="k", ls="--", label="connectome")
    ax.set_xscale("log")
    ax.set_xlabel("X (top DE genes per neuron)")
    ax.set_ylabel("network-wide group-mate score")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "analysis3_community_score_vs_X.png", dpi=150)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# I/O + orchestration
# ----------------------------------------------------------------------------
def read_gene_sets(path):
    p, sets = Path(path), {}
    if p.suffix.lower() == ".gmt":
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3:
                sets[f[0]] = {g for g in f[2:] if g}
    else:
        df = pd.read_csv(p)
        tf, tg = df.columns[:2]
        for name, sub in df.groupby(tf):
            sets[str(name)] = set(sub[tg].astype(str)) | {str(name)}
    return sets


def pipeline(edges, meta, counts, reg_files, out, Xs, n_perm, gsea_perm, k, seed):
    out.mkdir(parents=True, exist_ok=True)
    meta = meta.copy()
    meta["neuron"] = meta["neuron"].astype(str)
    if "class" in meta and not (set(counts.columns) & set(meta["neuron"])):
        meta = meta[meta["class"].isin(counts.columns)]
        counts = counts[meta["class"].values]
        counts.columns = meta["neuron"].values
    nodes = [n for n in meta["neuron"] if n in counts.columns]
    meta = meta.set_index("neuron").loc[nodes]
    types = meta["type"]
    counts = counts[nodes]
    counts = counts.loc[counts.sum(axis=1) > 0]
    print(f"{len(nodes)} neurons, {counts.shape[0]} genes, types: {types.value_counts().to_dict()}")

    conn = build_connectome(edges, types)
    print(f"connectome (undirected projection): {conn.number_of_edges()} edges")
    lfc = one_vs_all_log2fc(counts)

    sources = {"all_genes": None}
    reg_sets = {}
    for name, sets in reg_files.items():
        sources[f"regulon_{name}"] = set().union(*sets.values())
        reg_sets.update({f"{name}:{k_}": v for k_, v in sets.items()})
    if len(reg_files) > 1:
        sources["regulon_combined"] = set().union(*reg_sets.values())

    print("Analysis 1")
    a1, nets = analysis1(conn, lfc, types, sources, Xs, n_perm, out, seed)
    print("Analysis 2")
    if reg_sets:
        analysis2(counts, types, lfc, reg_sets, gsea_perm, out, seed)
    print("Analysis 3")
    a3 = analysis3(conn, nets, types, k, max(n_perm, 200), out, seed)
    print(a1.round(3).to_string(index=False))
    print(a3.round(3).to_string(index=False))
    print(f"Results written to {out}/")


def synthetic(seed=0, n=120, n_genes=3000):
    rng = np.random.default_rng(seed)
    types = pd.Series(rng.choice(TYPES, n, p=[.3, .3, .3, .1]), index=[f"N{i:03d}" for i in range(n)])
    genes = [f"g{i}" for i in range(n_genes)]
    base = rng.gamma(0.5, 2.0, n_genes)
    markers = {t: rng.choice(n_genes, 150, replace=False) for t in TYPES}
    rate = np.tile(base[:, None], (1, n))
    for j, t in enumerate(types):
        rate[markers[t], j] *= 5
        rate[rng.choice(n_genes, 40, replace=False), j] *= 3
    rate = rate / rate.sum(0) * 1e5
    counts = pd.DataFrame(rng.poisson(rate), index=genes, columns=types.index)
    codes = types.values
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < (0.10 if codes[i] == codes[j] else 0.03):
                edges.append((types.index[i], types.index[j], int(rng.integers(1, 6))))
    edges = pd.DataFrame(edges, columns=["pre", "post", "weight"])
    regs = {f"TF{i}": set(rng.choice(genes, 40, replace=False)) for i in range(25)}
    for i, t in enumerate(TYPES):       # a few type-informative regulons
        regs[f"TF_{t}"] = {genes[x] for x in rng.choice(markers[t], 40, replace=False)}
    meta = pd.DataFrame({"neuron": types.index, "type": types.values})
    return edges, meta, counts, {"demo": regs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="results")
        p.add_argument("--X", type=int, nargs="+", default=[5, 10, 25, 50, 100, 250])
        p.add_argument("--n-perm", type=int, default=100, help="edge-swap permutations")
        p.add_argument("--gsea-perm", type=int, default=500)
        p.add_argument("--communities", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        if name == "run":
            p.add_argument("--connectome", required=True)
            p.add_argument("--meta", required=True)
            p.add_argument("--expression", required=True)
            p.add_argument("--regulons", action="append", default=[], help="name=path (repeatable)")
    a = ap.parse_args()
    if a.cmd == "demo":
        edges, meta, counts, regs = synthetic(a.seed)
    else:
        edges = pd.read_csv(a.connectome)
        meta = pd.read_csv(a.meta)
        counts = pd.read_csv(a.expression, index_col=0)
        regs = {}
        for spec in a.regulons:
            nm, path = spec.split("=", 1)
            regs[nm] = read_gene_sets(path)
    pipeline(edges, meta, counts, regs, Path(a.out), a.X, a.n_perm, a.gsea_perm,
             a.communities, a.seed)


if __name__ == "__main__":
    main()