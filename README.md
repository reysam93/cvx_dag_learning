# Convex learning of non-negative DAGs
This repository includes the code associated with the paper "[Non-negative Weighted DAG Structure Learning]([https://arxiv.org/abs/2207.04747](https://arxiv.org/abs/2409.07880))", by Samuel Rey, Seyed S. Saboksayr, and Gonzalo Mateos in the `master` branch.

## Repository layout

```text
src/                    Core models and shared utilities
baselines/              Adapted baseline implementations
synthetic/
  scripts/              Synthetic-data experiments
  notebooks/            Synthetic-data analyses and figures
real/sachs/
  scripts/              Sachs scripts, when needed
  notebooks/            Sachs experiments and data preparation
utilities/
  scripts/              Hyperparameter tuning and diagnostics
  notebooks/            Short tuning and exploratory notebooks
datasets/               Input datasets
results/                Experiment outputs
scripts/                Environment and installation helpers
```

Each adapted file in `baselines/` identifies its original repository. External
reference implementations may be cloned under `code_aux/`, which is ignored by
Git.

Run experiment scripts directly from the repository or by absolute path. They
resolve the project root automatically:

```bash
python synthetic/scripts/preliminary_exp.py
python synthetic/scripts/number_samples.py
python synthetic/scripts/graph_size.py
python utilities/scripts/hyperparameter_tuning.py
python utilities/scripts/diagnostics_run_algorithms.py --quick
```

The notebooks start with a project-root setup cell so their existing imports
and result paths continue to work from the new directories.

## Abstract
We address the problem of learning the topology of directed acyclic graphs (DAGs) from nodal observations, which adhere to a linear structural equation model. Recent advances framed the combinatorial DAG structure learning task as a continuous optimization problem, yet existing methods must contend with the complexities of non-convex optimization. To overcome this limitation, we assume that the latent DAG contains only non-negative edge weights. Leveraging this additional structure, we argue that cycles can be effectively characterized (and prevented) using a convex acyclicity function based on the log-determinant of the adjacency matrix. This convexity allows us to relax the task of learning the non-negative weighted DAG as an abstract convex optimization problem. We propose a DAG recovery algorithm based on the method of multipliers, that is guaranteed to return a global minimizer. Furthermore, we prove that in the infinite sample size regime, the convexity of our approach ensures the recovery of the true DAG structure. We empirically validate the performance of our algorithm in several reproducible synthetic-data test cases, showing that it outperforms state-of-the-art alternatives.
