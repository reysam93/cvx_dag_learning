import numpy as np
from numpy import linalg as la
import networkx as nx
import time
from pandas import DataFrame
from IPython.display import display

import matplotlib.pyplot as plt

def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def create_dag(n_nodes, graph_type, edges, permute=True, edge_type='positive', w_range=(.5, 1.5),
               rew_prob=.1):
    """
    edge_type cana be binary, positive, or negative 
    """    
    if 'er' in graph_type:
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.triu(nx.to_numpy_array(G), k=1)

    elif graph_type == 'sf' or graph_type == 'sf_t':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sf' else np.tril(adj, k=-1)

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sw' else np.tril(adj, k=-1)

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G) == False
    assert nx.is_empty(G) == False
    
    if permute:
        P = np.eye(n_nodes)
        P = P[:, np.random.permutation(n_nodes)]
        W = P @ W @ P.T

    if edge_type == 'binary':
        W_weighted = W.copy()
    elif edge_type == 'positive':
        weights = np.random.uniform(w_range[0], w_range[1], size=W.shape)
        W_weighted = weights * W
    elif edge_type == 'weighted':
        # Default range: w_range=((-2.0, -0.5), (0.5, 2.0))
        W_weighted = np.zeros(W.shape)
        S = np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights
    else:
        raise ValueError('Unknown edge type')

    dag = nx.DiGraph(W_weighted)
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"

    return W_weighted, nx.DiGraph(dag)


def create_sem_signals(n_nodes, n_samples, G, noise_type='normal', var=1):
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == n_nodes
    
    X = np.zeros((n_samples, n_nodes))

    W_weighted = nx.to_numpy_array(G)

    if np.isscalar(var):
        var = var*np.ones(n_nodes)
    
    # Perform X = Z(I-A)^-1 sequentially to increase speed
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W_weighted[parents, j])
        if noise_type == 'normal':
            scale = np.sqrt(var[j])
            X[:, j] = eta + np.random.normal(scale=scale, size=(n_samples))
        elif noise_type == 'exp':
            scale = np.sqrt(var[j])
            X[:, j] = eta + np.random.exponential(scale=scale, size=(n_samples))
        elif noise_type == 'laplace':
            scale = np.sqrt(var[j] / 2.0)
            X[:, j] = eta + np.random.laplace(loc=0.0, scale=scale, size=(n_samples))
        elif noise_type == 'gumbel':
            scale = np.sqrt(6.0 * var[j]) / np.pi
            X[:, j] = eta + np.random.gumbel(loc=0.0, scale=scale, size=(n_samples))
        else:
            raise ValueError('Noise type error!')

    return X


def simulate_sem(n_nodes, n_samples, graph_type, edges, permute=True, edge_type='positive',
                 w_range=(.5, 1.5), noise_type='normal', var=1):
    A, dag = create_dag(n_nodes, graph_type, edges, permute, edge_type, w_range)
    X = create_sem_signals(n_nodes, n_samples, dag, noise_type, var)
    return A, dag, X

def to_bin(W, thr=0.1):
    W_bin = np.copy(W)
    W_bin[np.abs(W_bin) < thr] = 0
    W_bin[np.abs(W_bin) >= thr] = 1

    return W_bin

def compute_norm_sq_err(W_true, W_est, norm_W_true=None):
    norm_W_true = norm_W_true if norm_W_true is not None else la.norm(W_true)
    norm_W_est = la.norm(W_est) if la.norm(W_est) > 0 else 1
    return (la.norm(W_true/norm_W_true - W_est/norm_W_est))**2

def _as_binary_graph(W):
    """Return a binary adjacency matrix, expanding -1 entries as undirected edges."""
    W = np.asarray(W)
    A = (W != 0).astype(int)
    undirected = np.argwhere(W == -1)
    for i, j in undirected:
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A

def _has_undirected_edges(A):
    return np.any(np.triu((A != 0) & (A.T != 0), k=1))

def _skeleton_edges(A):
    S = (A != 0) | (A.T != 0)
    return [(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[0]) if S[i, j]]

def _path_matrix(A):
    reach = A.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(A.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach

def _v_structures(A):
    """Return unshielded colliders a -> b <- c in a directed graph or CPDAG."""
    A = A.astype(bool)
    S = A | A.T
    v_structs = set()
    for b in range(A.shape[0]):
        parents = np.flatnonzero(A[:, b] & ~A[b, :])
        for idx, a in enumerate(parents):
            for c in parents[idx + 1:]:
                if not S[a, c]:
                    v_structs.add((min(a, c), b, max(a, c)))
    return v_structs

def _is_d_separator(A, x, y, z):
    """D-separation test via ancestral moralization."""
    z = set(z)
    if x in z or y in z:
        return False

    reach = _path_matrix(A)
    nodes = set(z) | {x, y}
    ancestral = np.any(reach[:, list(nodes)], axis=1)
    ancestral_nodes = np.flatnonzero(ancestral)
    idx = {node: pos for pos, node in enumerate(ancestral_nodes)}

    sub = A[np.ix_(ancestral_nodes, ancestral_nodes)].astype(bool)
    moral = sub | sub.T
    for child in range(sub.shape[0]):
        parents = np.flatnonzero(sub[:, child])
        for i, parent_i in enumerate(parents):
            for parent_j in parents[i + 1:]:
                moral[parent_i, parent_j] = True
                moral[parent_j, parent_i] = True

    if x not in idx or y not in idx:
        return True

    blocked = {idx[node] for node in z if node in idx}
    start = idx[x]
    target = idx[y]
    if start in blocked or target in blocked:
        return False

    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return False
        neighbors = np.flatnonzero(moral[node])
        for neighbor in neighbors:
            if neighbor not in seen and neighbor not in blocked:
                seen.add(neighbor)
                stack.append(neighbor)

    return True

def _is_valid_parent_adjustment(true_dag, intervention, outcome, adjustment):
    """Check whether adjustment is valid for intervention -> outcome in true_dag."""
    adjustment = set(adjustment)
    if intervention in adjustment or outcome in adjustment:
        return False

    reach = _path_matrix(true_dag)
    descendants = set(np.flatnonzero(reach[intervention])) - {intervention}
    if adjustment & descendants:
        return False

    backdoor_graph = true_dag.copy()
    backdoor_graph[intervention, :] = 0
    return _is_d_separator(backdoor_graph, intervention, outcome, adjustment)

def structural_intervention_distance(W_bin_true, W_bin_est):
    """Compute SID between a true DAG and an estimated DAG.

    The implementation counts ordered node pairs for which the parent adjustment
    set implied by the estimated graph is not valid in the true graph.
    """
    true_dag = _as_binary_graph(W_bin_true)
    est_dag = _as_binary_graph(W_bin_est)

    if _has_undirected_edges(true_dag) or not is_dag(true_dag):
        raise ValueError("W_bin_true must be a DAG.")
    if _has_undirected_edges(est_dag) or not is_dag(est_dag):
        raise ValueError("W_bin_est must be a DAG for SID. Use sid_c for CPDAGs.")

    n_nodes = true_dag.shape[0]
    sid = 0
    incorrect = np.zeros((n_nodes, n_nodes), dtype=int)
    parents_est = [set(np.flatnonzero(est_dag[:, i])) for i in range(n_nodes)]
    reach_true = _path_matrix(true_dag)
    reach_est = _path_matrix(est_dag)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if not reach_est[i, j]:
                is_correct = not reach_true[i, j]
            else:
                is_correct = _is_valid_parent_adjustment(true_dag, i, j, parents_est[i])

            if not is_correct:
                incorrect[i, j] = 1
                sid += 1

    return sid, incorrect

def sid(W_bin_true, W_bin_est, normalize=False):
    """Compute Structural Intervention Distance (SID) for two DAGs."""
    value, _ = structural_intervention_distance(W_bin_true, W_bin_est)
    if normalize:
        n_nodes = np.asarray(W_bin_true).shape[0]
        return value / n_nodes
    return value

def _enumerate_mec_dags_from_dag(dag, max_edges=20):
    edges = _skeleton_edges(dag)
    if len(edges) > max_edges:
        raise ValueError(
            f"SID-C enumerates Markov-equivalent DAGs; skeleton has {len(edges)} "
            f"edges, above max_edges={max_edges}."
        )

    target_v_structures = _v_structures(dag)
    n_nodes = dag.shape[0]
    for mask in range(1 << len(edges)):
        candidate = np.zeros((n_nodes, n_nodes), dtype=int)
        for bit, (i, j) in enumerate(edges):
            if (mask >> bit) & 1:
                candidate[i, j] = 1
            else:
                candidate[j, i] = 1
        if is_dag(candidate) and _v_structures(candidate) == target_v_structures:
            yield candidate

def _enumerate_cpdag_extensions(cpdag, max_undirected_edges=20):
    n_nodes = cpdag.shape[0]
    directed = [
        (i, j)
        for i in range(n_nodes)
        for j in range(n_nodes)
        if i != j and cpdag[i, j] and not cpdag[j, i]
    ]
    undirected = [
        (i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if cpdag[i, j] and cpdag[j, i]
    ]
    if len(undirected) > max_undirected_edges:
        raise ValueError(
            f"SID-C enumerates CPDAG extensions; CPDAG has {len(undirected)} "
            f"undirected edges, above max_undirected_edges={max_undirected_edges}."
        )

    target_v_structures = _v_structures(cpdag)
    for mask in range(1 << len(undirected)):
        candidate = np.zeros((n_nodes, n_nodes), dtype=int)
        for i, j in directed:
            candidate[i, j] = 1
        for bit, (i, j) in enumerate(undirected):
            if (mask >> bit) & 1:
                candidate[i, j] = 1
            else:
                candidate[j, i] = 1
        if is_dag(candidate) and _v_structures(candidate) == target_v_structures:
            yield candidate

def dag_to_cpdag(W_bin, max_edges=20):
    """Map a DAG to its CPDAG by enumerating its Markov equivalence class."""
    dag = _as_binary_graph(W_bin)
    if _has_undirected_edges(dag) or not is_dag(dag):
        raise ValueError("W_bin must be a DAG.")

    mec_dags = list(_enumerate_mec_dags_from_dag(dag, max_edges=max_edges))
    if not mec_dags:
        raise ValueError("Could not enumerate a Markov equivalence class for W_bin.")

    cpdag = np.zeros_like(dag)
    for i, j in _skeleton_edges(dag):
        i_to_j = [candidate[i, j] == 1 for candidate in mec_dags]
        if all(i_to_j):
            cpdag[i, j] = 1
        elif not any(i_to_j):
            cpdag[j, i] = 1
        else:
            cpdag[i, j] = 1
            cpdag[j, i] = 1

    return cpdag

def _cpdag_edge_state(cpdag, i, j):
    if cpdag[i, j] and cpdag[j, i]:
        return "undirected"
    if cpdag[i, j]:
        return "i_to_j"
    if cpdag[j, i]:
        return "j_to_i"
    return "none"

def shd_c(W_bin_true, W_bin_est, max_edges=20):
    """Compute SHD-C: SHD after mapping true and estimated DAGs to CPDAGs."""
    true_cpdag = dag_to_cpdag(W_bin_true, max_edges=max_edges)
    est_cpdag = dag_to_cpdag(W_bin_est, max_edges=max_edges)

    shd = 0
    for i in range(true_cpdag.shape[0]):
        for j in range(i + 1, true_cpdag.shape[0]):
            if _cpdag_edge_state(true_cpdag, i, j) != _cpdag_edge_state(est_cpdag, i, j):
                shd += 1

    return shd

def sid_c(W_bin_true, W_bin_est, normalize=False, max_edges=20):
    """Compute SID-C bounds between a true DAG and an estimated DAG/CPDAG.

    If W_bin_est is a DAG, it is evaluated over its Markov equivalence class. If
    it is a CPDAG, all consistent DAG extensions are considered. The return value
    is (lower_bound, upper_bound), matching the best and worst DAG in that class.
    """
    true_dag = _as_binary_graph(W_bin_true)
    est_graph = _as_binary_graph(W_bin_est)

    if _has_undirected_edges(true_dag) or not is_dag(true_dag):
        raise ValueError("W_bin_true must be a DAG.")

    if _has_undirected_edges(est_graph):
        candidates = list(_enumerate_cpdag_extensions(est_graph, max_undirected_edges=max_edges))
    else:
        if not is_dag(est_graph):
            raise ValueError("W_bin_est must be a DAG or a CPDAG.")
        candidates = list(_enumerate_mec_dags_from_dag(est_graph, max_edges=max_edges))

    if not candidates:
        raise ValueError("Could not enumerate any DAGs for SID-C.")

    values = [structural_intervention_distance(true_dag, candidate)[0] for candidate in candidates]
    lower = min(values)
    upper = max(values)
    if normalize:
        denom = true_dag.shape[0] * (true_dag.shape[0] - 1)
        lower /= denom
        upper /= denom

    return lower, upper

def count_accuracy(W_bin_true, W_bin_est, compute_sid=False, sid_normalize=False,
                   compute_sid_c=False, sid_c_normalize=False, max_sid_c_edges=20,
                   compute_shd_c=False, max_shd_c_edges=20):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.
        compute_sid (bool): if True, also return SID.
        sid_normalize (bool): if True, divide SID by d.
        compute_sid_c (bool): if True, also return SID-C lower and upper bounds.
        sid_c_normalize (bool): if True, divide SID-C by d * (d - 1).
        max_sid_c_edges (int): maximum number of edges to enumerate for SID-C.
        compute_shd_c (bool): if True, also return SHD-C.
        max_shd_c_edges (int): maximum number of edges to enumerate for SHD-C.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.
        sid: returned only if compute_sid is True.
        sid_c_lower, sid_c_upper: returned only if compute_sid_c is True.
        shd_c: returned only if compute_shd_c is True.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    pred_und = np.flatnonzero(W_bin_est == -1)
    pred = np.flatnonzero(W_bin_est == 1)
    cond = np.flatnonzero(W_bin_true)
    cond_reversed = np.flatnonzero(W_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # Compute SHD
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(W_bin_est + W_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(W_bin_true + W_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Compute TPR
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    tpr = float(len(true_pos)) / max(len(cond), 1)

    # Compute FDR
    pred_size = len(pred) + len(pred_und)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)

    out = [shd, tpr, fdr]

    if compute_sid:
        out.append(sid(W_bin_true, W_bin_est, normalize=sid_normalize))

    if compute_sid_c:
        sid_c_lower, sid_c_upper = sid_c(
            W_bin_true,
            W_bin_est,
            normalize=sid_c_normalize,
            max_edges=max_sid_c_edges,
        )
        out.extend([sid_c_lower, sid_c_upper])

    if compute_shd_c:
        out.append(shd_c(W_bin_true, W_bin_est, max_edges=max_shd_c_edges))

    return tuple(out)

def display_results(exps_leg, metrics, agg='mean', file_name=None):
    
    metric_str = {'leg': exps_leg}
    for key, value in metrics.items():
        metric_str[key] = []
        
        agg_metric = np.median(value, axis=0) if agg == 'median' else np.mean(value, axis=0)
        std_metric = np.std(value, axis=0)
        for i, _ in enumerate(exps_leg):
            text = f'{agg_metric[i]:.4f}  \u00B1 {std_metric[i]:.4f}'
            metric_str[key].append(text)
        
    df = DataFrame(metric_str)
    display(df)

    if file_name:
        df.to_csv(f'{file_name}.csv', index=False)
        print(f'DataFrame saved to {file_name}.csv')

def standarize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)

def plot_data(axes, data, exps, x_vals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=None,
              alpha=.25, plot_func='semilogx'):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    std = np.std(data, axis=0)
    prctile25 = np.percentile(data, 25, axis=0)
    prctile75 = np.percentile(data, 75, axis=0)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        getattr(axes, plot_func)(x_vals, agg_data[:,i], exp['fmt'], label=exp['leg'])

        if deviation == 'prctile':
            up_ci = prctile25[:,i]
            low_ci = prctile75[:,i]
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)
        elif deviation == 'std':
            up_ci = agg_data[:,i] + std[:,i]
            low_ci = np.maximum(agg_data[:,i] - std[:,i], 0)
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()

def plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, x_vals, exps, 
                     agg='mean', skip_idx=[], dev=False, alpha=.25, xlabel='Number of samples'):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], shd, exps, x_vals, xlabel, 'SDH', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[1], tpr, exps, x_vals, xlabel, 'TPR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[2], fdr, exps, x_vals, xlabel, 'FDR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[3], fscore, exps, x_vals, xlabel, 'F1', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], err, exps, x_vals, xlabel, 'Fro Error', skip_idx, agg=agg,
              deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[1], acyc, exps, x_vals, xlabel, 'Acyclity', skip_idx, agg=agg,
              deviation=dev)
    plot_data(axes[2], runtime, exps, x_vals, xlabel, 'Running time (seconds)',
              skip_idx, agg=agg, deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[3], dag_count, exps, x_vals, xlabel, 'Graph is DAG', skip_idx,
              agg=agg)
    plt.tight_layout()


def data_to_csv(fname, models, xaxis, error, agg='mean', dev='std'):
    data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error), axis=1)
    header = 'xaxis; '  

    for i, model in enumerate(models):
        header += model['leg']
        if i < len(models)-1:
            header += '; '

    np.savetxt(fname, data, delimiter=';', header=header, comments='')
    print('SAVED as:', fname)
