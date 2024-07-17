import numpy as np
from numpy import linalg as la
import networkx as nx
import time
from pandas import DataFrame
from IPython.display import display

import matplotlib.pyplot as plt

def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def create_dag(n_nodes, graph_type, edges, permute=True, edge_type='positive', w_range=(.5, 1.5)):
    """
    edge_type cana be binary, positive, or negative 
    """    
    if 'er' in graph_type:
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.triu(nx.to_numpy_array(G), k=1)

        if permute:
            P = np.eye(n_nodes)
            P = P[:, np.random.permutation(n_nodes)]
            W = P @ W @ P.T

    elif graph_type == 'sf':
        # NOTE: Why are this graphs lower triangular?
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        # W = np.tril(adj, k=-1)
        W = np.triu(adj, k=1)

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G)==False
    assert nx.is_empty(G)==False
    
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

    # Perform X = Z(I-A)^-1 sequentially to increase speed
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W_weighted[parents, j])
        if noise_type == 'normal':
            scale = np.sqrt(var)
            X[:, j] = eta + np.random.normal(scale=scale, size=(n_samples))
        elif noise_type == 'exp':
            scale = np.sqrt(var)
            X[:, j] = eta + np.random.exponential(scale=scale, size=(n_samples))
        elif noise_type == 'laplace':
            scale = np.sqrt(var / 2.0)
            X[:, j] = eta + np.random.laplace(loc=0.0, scale=scale, size=(n_samples))
        elif noise_type == 'gumbel':
            scale = np.sqrt(6.0 * var) / np.pi
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

def count_accuracy(W_bin_true, W_bin_est):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

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

    return shd, tpr, fdr

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

def plot_data(axes, data, exps, x_vals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=False,
              alpha=.25, plot_func='semilogx'):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    std = np.std(data, axis=0)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        getattr(axes, plot_func)(x_vals, agg_data[:,i], exp['fmt'], label=exp['leg'])

        if deviation:
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