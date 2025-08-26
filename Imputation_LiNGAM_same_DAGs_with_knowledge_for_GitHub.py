# ============================================================
# MultiGroupDirectLiNGAM + Hot-deck multiple imputation
# ============================================================
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from alive_progress import alive_bar
from lingam import MultiGroupDirectLiNGAM
from lingam.utils import make_prior_knowledge
import graphviz

# ---------------- helpers ----------------
def get_xvars(df: pd.DataFrame):
    """Detect X1 … Xk automatically (keep Z, Y out)."""
    return [c for c in df.columns if c.upper() not in ("Z","Y")]

def build_pools(df: pd.DataFrame, strata_col="Z", vars_for_pool=None):
    """Donor pools per Z-level and variable for hot-deck."""
    if vars_for_pool is None:
        vars_for_pool = get_xvars(df) + ["Y"]
    pools = {}
    for z, g in df.groupby(strata_col):
        for v in vars_for_pool:
            pools[(z, v)] = g[v].dropna().to_numpy()
    return pools

def impute_hotdeck_once(df: pd.DataFrame, pools, rng=None, strata_col="Z"):
    """One hot-deck completed frame (within-Z random donor, with replacement)."""
    if rng is None:
        rng = np.random.default_rng()
    di = df.copy(deep=True)
    targets = get_xvars(df) + ["Y"]
    for i, row in di.iterrows():
        z = row[strata_col]
        for v in targets:
            if pd.isna(row[v]):
                pool = pools.get((z, v), np.array([]))
                if pool.size:
                    di.at[i, v] = rng.choice(pool, 1)[0]
                else:
                    all_pool = di[v].dropna().to_numpy()
                    if all_pool.size:
                        di.at[i, v] = rng.choice(all_pool, 1)[0]
    return di

def make_imputation_list(bootstrap_df: pd.DataFrame, n_repeats: int, seed_base: int):
    """Return list of n_repeats completed arrays (no vertical stacking!)."""
    pools = build_pools(bootstrap_df, strata_col="Z")
    X_list = []
    for r in range(n_repeats):
        rng  = np.random.default_rng(seed_base + r)
        comp = impute_hotdeck_once(bootstrap_df, pools, rng=rng, strata_col="Z")
        X_list.append(comp[labels].astype(float).to_numpy())
    return X_list

def to_key_from_binary(bin_mat: np.ndarray) -> str:
    """0/1 adjacency → row-major bit string (child=row, parent=col)."""
    return ''.join(map(str, bin_mat.astype(int).flatten(order="C")))

def draw_dag_from_adj(A: np.ndarray, labels, fname="dag_top"):
    """Draw weighted DAG (parent=column, child=row)."""
    g = graphviz.Digraph(engine="dot", graph_attr={"rankdir":"LR"})
    for lab in labels:
        g.node(lab, shape="box")
    k = len(labels)
    for j in range(k):          # parent=col=j
        for i in range(k):      # child =row=i
            w = A[i, j]
            if abs(w) > 1e-9:
                colour = "black" if w > 0 else "red"
                g.edge(labels[j], labels[i], label=f"{w:.2f}",
                       color=colour, penwidth="2")
    g.render(fname, format="pdf")

def chunk_key(s: str, k: int):
    """Pretty block print: k chunks of k (for readability)."""
    return ' '.join(s[i:i+k] for i in range(0, k*k, k))

# ---- DFS to enumerate all directed paths from src→dst in a binary DAG
def enumerate_paths(bin_mat, w_mat, src, dst, max_len=None):
    k = bin_mat.shape[0]
    if max_len is None:
        max_len = k
    stack = [(src, [src], 1.0)]
    while stack:
        node, path, eff = stack.pop()
        if len(path) > max_len:
            continue
        if node == dst:
            yield path, eff
            continue
        children = np.where(bin_mat[:, node] == 1)[0]  # node → child
        for ch in children:
            if ch in path:    # avoid cycles
                continue
            w = w_mat[ch, node]
            stack.append((ch, path + [ch], eff * w))

# ---------------- data -------------------
dat = pd.read_csv("Missing_all.csv")
dat.columns = ["Z","X1","X2","X3","X4","X5","Y"]

labels  = ["Z"] + get_xvars(dat) + ["Y"]   # ["Z","X1",...,"X5","Y"]
k       = len(labels)

# ---------------- params -----------------
n_sampling = 5000     # bootstrap samples (outer loop)
n_repeats  = 50      # imputations per bootstrap sample
thr        = 1e-4     # threshold to call an edge present (|w|>thr)
seed       = 2025_0819

# prior knowledge: Z exogenous, Y sink
prior_k = make_prior_knowledge(
    n_variables = k,
    exogenous_variables = [0],     # Z
    sink_variables      = [k-1],   # Y
)

# ---------------- tally ( imputation ×  bootstrap ) -------------
dag_counter    = Counter()                               # total DAGs
edge_counter   = np.zeros((k, k), dtype=int)
sum_adj_by_key = defaultdict(lambda: np.zeros((k, k)))
count_by_key   = Counter()

# Z→Y path
path_counter     = Counter()
path_effect_sum  = defaultdict(float)

rng_outer = np.random.default_rng(seed)
src = labels.index("Z")
dst = labels.index("Y")

# ---------------- main -------------------
with alive_bar(n_sampling, title="MultiGroup LiNGAM (hot-deck, per-imputation tally)") as bar:
    for b in range(n_sampling):
        # (1) bootstrap the rows (with replacement)
        idx = rng_outer.integers(0, len(dat), size=len(dat))
        boot_df = dat.iloc[idx].reset_index(drop=True)

        # (2) build n_repeats imputed datasets (list of arrays)
        X_list = make_imputation_list(boot_df, n_repeats, seed_base=10_000*b)

        # (3) fit MultiGroupDirectLiNGAM with prior knowledge
        model = MultiGroupDirectLiNGAM(prior_knowledge=prior_k, random_state=777+b)
        model.fit(X_list)

        # (4) Count each DAG
        adjs = np.asarray(model.adjacency_matrices_)   # (n_repeats, k, k)
        for r in range(n_repeats):
            A = adjs[r]
            bin_ = (np.abs(A) > thr).astype(int)

            # DAG strings
            key = to_key_from_binary(bin_)
            dag_counter[key]     += 1
            edge_counter         += bin_
            sum_adj_by_key[key]  += A
            count_by_key[key]    += 1

            # Z→Y path
            for path_nodes, total_eff in enumerate_paths(bin_, A, src, dst):
                path_str = "→".join(labels[i] for i in path_nodes)
                path_counter[path_str]    += 1
                path_effect_sum[path_str] += float(total_eff)

        bar()

# ---------------- results: DAG / Edge ----------------
total_dags = n_sampling * n_repeats
dag_prob  = {k_: v/total_dags for k_, v in dag_counter.items()}
edge_prob = edge_counter / total_dags

dag_df = (pd.DataFrame.from_dict(dag_prob, orient="index", columns=["probability"])
          .sort_values("probability", ascending=False))
edge_df = pd.DataFrame(edge_prob, index=labels, columns=labels)

print("\nTop-10 DAGs (over ALL imputations × bootstraps):")
def chunk_key(s: str, k: int):
    return ' '.join(s[i:i+k] for i in range(0, k*k, k))
for k_, p in dag_df.head(10)["probability"].items():
    print(f"{chunk_key(k_, k)}    {p:.4f}")

print("\nEdge probabilities (parent → child, over ALL DAGs):")
print(edge_df.round(3))

# The most frequent DAG
top_key   = dag_df.index[0]
top_prob  = dag_df.iloc[0, 0]
mean_adj_top = sum_adj_by_key[top_key] / count_by_key[top_key]
print(f"\nMost frequent DAG key = {chunk_key(top_key, k)}  (p={top_prob:.3f})")
print("Mean adjacency of the most frequent DAG:")
print(pd.DataFrame(mean_adj_top, index=labels, columns=labels).round(3))
draw_dag_from_adj(mean_adj_top, labels, fname="dag_top_multigroup_hotdeck_per_imputation")
print("\nSaved DAG as  dag_top_multigroup_hotdeck_per_imputation_20250820.pdf")

# ---------------- results: PATH(Z→Y) level ----------------
rows = []
for p, cnt in path_counter.items():
    prob    = cnt / total_dags
    ave_eff = path_effect_sum[p] / cnt
    rows.append(dict(Path=p, Probability=prob, MeanTotalEffect=ave_eff))

path_df = (pd.DataFrame(rows)
           .sort_values(["Probability", "Path"], ascending=[False, True])
           .reset_index(drop=True))
# out_csv = "Z_to_Y_path_stats_HotDeck_per_imputation_20250820.csv"
path_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"\n=== Z → Y path statistics (over ALL DAGs) ===")
print(path_df.head(20).round(3))
print(f"\nSaved path table to  {out_csv}")

