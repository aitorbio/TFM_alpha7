#!/usr/bin/env bash
# setup_environment.sh
# Instalación completa del entorno tfm_alpha7
# Pipeline SBDD — Moduladores alostéricos del α7 nAChR
#
# Uso:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#


set -e   # detener si cualquier comando falla

ENV_NAME="tfm_alpha7"
PYTHON_VERSION="3.10"

echo ""
echo "=================================================================="
echo "  Instalación del entorno: $ENV_NAME"
echo "  Pipeline SBDD α7 nAChR"
echo "=================================================================="
echo ""

# ── 1. Verificar que conda está disponible ────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda no encontrado."
    echo "Instala Miniconda desde: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ conda encontrado: $(conda --version)"


# ── 2. Crear entorno conda ────────────────────────────────────────────────────
echo ""
echo "── Creando entorno conda '$ENV_NAME' (Python $PYTHON_VERSION)..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  El entorno '$ENV_NAME' ya existe."
    echo "  Para reinstalar desde cero: conda env remove -n $ENV_NAME"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "  ✓ Entorno creado"
fi


# ── 3. Activar entorno ────────────────────────────────────────────────────────
echo ""
echo "── Activando entorno '$ENV_NAME'..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "  ✓ Entorno activado"


# ── 4. Instalar paquetes conda-forge ─────────────────────────────────────────
echo ""
echo "── Instalando paquetes científicos desde conda-forge..."
echo "   (rdkit, openbabel, pandas, numpy, scipy, scikit-learn, matplotlib)"

conda install -c conda-forge -y \
    "rdkit=2023.09" \
    "openbabel=3.1.1" \
    "pandas>=1.5" \
    "numpy>=1.24" \
    "scipy>=1.9" \
    "scikit-learn>=1.1" \
    "matplotlib>=3.6" \
    "seaborn" \
    "requests" \
    "urllib3" \
    "tqdm"

echo "  ✓ Paquetes conda-forge instalados"


# ── 5. Instalar AutoDock Vina ─────────────────────────────────────────────────
echo ""
echo "── Instalando AutoDock Vina 1.2.x..."

# Intentar desde bioconda primero, luego conda-forge
if conda install -c bioconda -c conda-forge -y autodock-vina 2>/dev/null; then
    echo "  ✓ AutoDock Vina instalado (bioconda)"
elif conda install -c conda-forge -y autodock-vina 2>/dev/null; then
    echo "  ✓ AutoDock Vina instalado (conda-forge)"
else
    echo "  ⚠ AutoDock Vina no disponible en conda."
    echo "    Descarga manual desde:"
    echo "    https://github.com/ccsb-scripps/AutoDock-Vina/releases"
    echo "    Descarga 'vina_1.2.5_linux_x86_64' (Linux) o el equivalente para tu SO"
    echo "    y colócalo en: $CONDA_PREFIX/bin/vina"
    echo "    chmod +x $CONDA_PREFIX/bin/vina"
fi


# ── 6. Instalar paquetes pip ───────────────────────────────────────────────────
echo ""
echo "── Instalando paquetes pip (ProLIF, MDAnalysis, mdtraj)..."

pip install --quiet \
    "prolif>=1.1.0" \
    "MDAnalysis>=2.4" \
    "mdtraj"

echo "  ✓ Paquetes pip instalados"


# ── 7. Verificación de la instalación ────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Verificación de la instalación"
echo "=================================================================="

ALL_OK=true

check_python() {
    local import_str="$1"
    local label="$2"
    if python -c "$import_str" &>/dev/null; then
        VER=$(python -c "$import_str; import sys" 2>/dev/null || echo "?")
        echo "  ✓ $label"
    else
        echo "  ✗ $label  ← FALTA"
        ALL_OK=false
    fi
}

check_tool() {
    local cmd="$1"
    local label="$2"
    local test_arg="${3:---version}"
    if command -v "$cmd" &>/dev/null; then
        VER=$($cmd $test_arg 2>&1 | head -1)
        echo "  ✓ $label  ($VER)"
    else
        echo "  ✗ $label  ← FALTA"
        ALL_OK=false
    fi
}

echo ""
echo "── Librerías Python ─────────────────────────────────────────────"
check_python "from rdkit import Chem; from rdkit import __version__ as v; print(v)"   "RDKit"
check_python "import pandas as pd; print(pd.__version__)"                              "pandas"
check_python "import numpy as np; print(np.__version__)"                               "numpy"
check_python "import scipy; print(scipy.__version__)"                                  "scipy"
check_python "import sklearn; print(sklearn.__version__)"                              "scikit-learn"
check_python "import matplotlib; print(matplotlib.__version__)"                        "matplotlib"
check_python "import requests; print(requests.__version__)"                            "requests"
check_python "import tqdm; print(tqdm.__version__)"                                    "tqdm"
check_python "import prolif; print(prolif.__version__)"                                "ProLIF"
check_python "import MDAnalysis; print(MDAnalysis.__version__)"                        "MDAnalysis"

echo ""
echo "── Herramientas de línea de comandos ────────────────────────────"
check_tool "obabel"  "OpenBabel"    "--version"
check_tool "vina"    "AutoDock Vina" "--version"

echo ""
if $ALL_OK; then
    echo "=================================================================="
    echo "  ✓ Instalación completa. Todos los componentes disponibles."
    echo "=================================================================="
else
    echo "=================================================================="
    echo "  ⚠ Algunos componentes faltan. Revisa los mensajes anteriores."
    echo "=================================================================="
fi

echo ""
echo "── Próximos pasos ───────────────────────────────────────────────"
echo "  1. Activa el entorno en cada sesión:"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. Ejecuta el test del módulo 1:"
echo "     python scripts/mol_properties_rdkit.py --test"
echo ""
echo "  3. Sigue la guía de ejecución paso a paso."
echo "=================================================================="
