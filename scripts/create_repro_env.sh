#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv-dag}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
BUILD_DIR="${BUILD_DIR:-${ENV_DIR}/src}"
DAGUERREOTYPE_DIR="${DAGUERREOTYPE_DIR:-${BUILD_DIR}/DAGuerreotype}"
LP_SPARSEMAP_REPO="${LP_SPARSEMAP_REPO:-https://github.com/deep-spin/lp-sparsemap.git}"
DAGUERREOTYPE_REPO="${DAGUERREOTYPE_REPO:-https://github.com/vzantedeschi/DAGuerreotype.git}"
EIGEN_URL="${EIGEN_URL:-https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable '${PYTHON_BIN}' was not found." >&2
    echo "Set PYTHON_BIN=/path/to/python3.10 or install Python 3.10." >&2
    exit 1
fi

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install -r requirements-repro.txt

mkdir -p "${BUILD_DIR}"

if [ ! -d "${DAGUERREOTYPE_DIR}" ]; then
    git clone "${DAGUERREOTYPE_REPO}" "${DAGUERREOTYPE_DIR}"
fi

if [ ! -d "${BUILD_DIR}/lp-sparsemap" ]; then
    git clone "${LP_SPARSEMAP_REPO}" "${BUILD_DIR}/lp-sparsemap"
fi

if [ ! -d "${BUILD_DIR}/eigen-master" ]; then
    curl -L "${EIGEN_URL}" -o "${BUILD_DIR}/eigen-master.zip"
    unzip -q "${BUILD_DIR}/eigen-master.zip" -d "${BUILD_DIR}"
fi

LP_SPARSEMAP_ROOT="$(cd "${BUILD_DIR}/lp-sparsemap" && pwd)"
EIGEN_ROOT="$(cd "${BUILD_DIR}/eigen-master" && pwd)"

if ! grep -q "EIGEN_ROOT = os.environ.get('EIGEN_ROOT'" "${LP_SPARSEMAP_ROOT}/setup.py"; then
    "${ENV_DIR}/bin/python" - <<'PY' "${LP_SPARSEMAP_ROOT}/setup.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("import sys\n", "import sys\nimport os\n")
text = text.replace(
    "EIGEN_URL = 'http://bitbucket.org/eigen/eigen/get/3.3.9.tar.gz'",
    "EIGEN_ROOT = os.environ.get('EIGEN_ROOT')\nEIGEN_URL = 'http://bitbucket.org/eigen/eigen/get/3.3.9.tar.gz'",
)
text = text.replace(
    "download_eigen()",
    "download_eigen() if EIGEN_ROOT is None else None",
)
text = text.replace(
    "'eigen-eigen-226a22f62e98',",
    "EIGEN_ROOT or 'eigen-eigen-226a22f62e98',",
)
path.write_text(text)
PY
fi

EIGEN_ROOT="${EIGEN_ROOT}" "${ENV_DIR}/bin/python" -m pip install --no-build-isolation -e "${LP_SPARSEMAP_ROOT}"

if ! grep -q "include_path=.*LP_SPARSEMAP_ROOT" "${DAGUERREOTYPE_DIR}/setup.py"; then
    "${ENV_DIR}/bin/python" - <<'PY' "${DAGUERREOTYPE_DIR}/setup.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "import os" not in text.splitlines()[:10]:
    text = text.replace("from setuptools import setup", "import os\nfrom setuptools import setup")
text = text.replace(
    "cythonize(extensions),",
    "cythonize(extensions, include_path=[os.environ['LP_SPARSEMAP_ROOT']]),",
)
path.write_text(text)
PY
fi

LP_SPARSEMAP_ROOT="${LP_SPARSEMAP_ROOT}" "${ENV_DIR}/bin/python" -m pip install --no-build-isolation -e "${DAGUERREOTYPE_DIR}"

"${ENV_DIR}/bin/python" -m ipykernel install --user \
    --name cvx-dag-learning \
    --display-name "cvx_dag_learning (.venv-dag)"

"${ENV_DIR}/bin/python" - <<'PY'
import numpy as np

from baselines.golem import GOLEM_EV
from baselines.sortnregress import VarSortNRegress
from baselines.daguerreotype import DAGuerreotype

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 3))

VarSortNRegress(alpha=0.01).fit(X)
GOLEM_EV(num_iter=1, checkpoint_iter=1).fit(X)
DAGuerreotype(num_epochs=1, num_inner_iters=1, verbose=False).fit(X)

print("Smoke tests passed.")
PY
