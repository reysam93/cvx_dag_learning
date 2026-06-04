# Reproducible Python Environment

This repository uses `.venv-dag` as a reproducible Python 3.10 environment for
the project code, the adapted baselines, and DAGuerreotype.

Create the environment:

```bash
bash scripts/create_repro_env.sh
```

Activate it:

```bash
source .venv-dag/bin/activate
```

Use it in notebooks by selecting the kernel:

```text
cvx_dag_learning (.venv-dag)
```

The Python dependencies are listed in `requirements-repro.txt`. The script also
downloads and builds the original DAGuerreotype `lp-sparsemap` dependency from:

- https://github.com/deep-spin/lp-sparsemap
- https://gitlab.com/libeigen/eigen

The setup script downloads and builds DAGuerreotype inside the virtual
environment source directory:

```text
.venv-dag/src/DAGuerreotype
```

Notes:

- The target Python version is 3.10.
- The environment intentionally uses `scikit-learn==1.1.1` to stay compatible
  with DAGuerreotype's original LARS edge-estimator path.
- Building DAGuerreotype requires system build tools such as `g++`, `make`,
  `curl`, `unzip`, and Python development headers for Python 3.10.
- R/SID packages from the original DAGuerreotype install script are not
  installed here because this repository already computes SID internally.
- Set `DAGUERREOTYPE_DIR=/path/to/DAGuerreotype` before running the script if
  you want to reuse an existing clone instead of downloading a fresh copy.
- The setup script applies two build-only compatibility patches: it makes
  `lp-sparsemap` accept the current Eigen headers, and it passes the
  `lp-sparsemap` source path to DAGuerreotype's Cython build so its `.pxd`
  files are found.
