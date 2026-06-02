"""
verify_installation.py
---

Verifica que todos los paquetes del pipeline están correctamente
instalados y son funcionales. Este script es una ayuda para verificar
que todo está instalado correctamente y permite replicar el pipline en
caso de que los profesores/tutor quieran reproducirlo. También pondre un
archivo ymal con la configuración del entorno y las versiones de las librerias.

Esta pensado para ser ejecutado en la terminal dentro del entorno virtual.

Uso:
conda activate tfm_alpha7
python scripts/verify_installation.py
"""

import sys

results = []


# Función helper para comprobar si un paquete está instalado y es funcional
def check(label, fn):
    try:
        version = fn()
        results.append((True, label, version))
    except Exception as e:
        results.append((False, label, str(e)[:60]))


# Comprobación de Python
check("Python", lambda: sys.version.split()[0])


# Comprobación de RDKit
def test_rdkit():
    from rdkit import Chem, __version__
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

    # Prueba funcional real con SMILES del propanoic acid
    mol = Chem.MolFromSmiles("CC(C)(C)OC(=O)NC(Cc1ccc(cc1)C(F)F)C(O)=O")
    assert mol is not None
    # ExactMolWt = peso molecular exacto (~315.31); MolWt = peso molecular promedio (~315)
    mw_exact = Descriptors.ExactMolWt(mol)
    mw_avg = Descriptors.MolWt(mol)
    assert 315 < mw_exact < 316, f"ExactMolWt inesperado: {mw_exact}"
    assert 305 < mw_avg < 325, f"MolWt inesperado: {mw_avg}"
    # Comprobar que PAINS funciona para evitar falsos positivos en cribado
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    cat = FilterCatalog.FilterCatalog(params)
    return __version__


check("RDKit (funcional)", test_rdkit)


# Comprobación de pandas
def test_pandas():
    import pandas as pd

    # Prueba del tamaño de los dataframes
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert len(df) == 2
    return pd.__version__


check("pandas", test_pandas)


# Comprobación de numpy
def test_numpy():
    import numpy as np

    # Prueab de la media de un array
    a = np.array([1.0, 2.0, 3.0])
    assert np.mean(a) == 2.0
    return np.__version__


check("numpy", test_numpy)


# Comprobación de scipy
def test_scipy():
    import scipy
    from scipy.stats import norm

    return scipy.__version__


check("scipy", test_scipy)


# Comprobación de scikit-learn
def test_sklearn():
    import sklearn
    from sklearn.metrics import roc_curve, auc
    import numpy as np

    # Prueba de la curva ROC
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(y, s)
    assert auc(fpr, tpr) > 0
    return sklearn.__version__


check("scikit-learn (funcional)", test_sklearn)


# Comprobación de matplotlib
def test_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prueba de la gráfica de línea
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.close()
    return matplotlib.__version__


check("matplotlib", test_matplotlib)


# Comprobación de requests
def test_requests():
    import requests

    return requests.__version__


check("requests", test_requests)


# Comprobación de tqdm
def test_tqdm():
    import tqdm

    return tqdm.__version__


check("tqdm", test_tqdm)


# Comprobación de OpenBabel (obabel CLI)
def test_obabel():
    import subprocess

    # Prueba del comando obabel --version
    r = subprocess.run(
        ["obabel", "--version"], capture_output=True, text=True, timeout=10
    )
    if r.returncode != 0 and "Open Babel" not in r.stdout + r.stderr:
        raise RuntimeError("obabel no responde")
    line = (r.stdout + r.stderr).split("\n")[0]
    return line.strip()[:40]


check("OpenBabel (CLI)", test_obabel)


# Comprobación de AutoDock Vina  - Hay que tener cuidado ya que puede que el
# sistema puede que reconozca AutoDockVina como malware.
def test_vina():
    import subprocess

    try:
        r = subprocess.run(
            ["vina", "--version"], capture_output=True, text=True, timeout=10
        )
        line = (r.stdout + r.stderr).split("\n")[0]
        if not line.strip():
            raise RuntimeError("sin salida")
        return line.strip()[:40]
    except FileNotFoundError:
        raise RuntimeError(
            # Instrucciones para instalar AutoDock Vina
            "No encontrado. Descarga desde:\n"
            "    https://github.com/ccsb-scripps/AutoDock-Vina\n"
            "    y coloca el programa en: ~/miniconda3/envs/tfm_alpha7/bin/vina"
        )


check("AutoDock Vina (CLI)", test_vina)


# Comprobación de ProLIF
def test_prolif():
    import prolif

    return prolif.__version__


check("ProLIF", test_prolif)


# Comprobación de MDAnalysis
def test_mda():
    import MDAnalysis

    return MDAnalysis.__version__


check("MDAnalysis", test_mda)


# Comprobación de mol_properties_rdkit.py
# mol_properties_rdkit.py calcula propiedades moleculares usando RDKit,
# es necesario que esté instalado y que funcione correctamente. Es esencial
# ya que se usa en el resto de scripts y es la base del cribado de compuestos.
def test_module():
    import importlib.util, sys, os

    script = "mol_properties_rdkit.py"
    # Buscar en el mismo directorio, imporatante que esté en el mismo directorio
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, script)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{script} no encontrado en {here}")
    spec = importlib.util.spec_from_file_location("mol_properties_rdkit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mol_properties_rdkit"] = (
        mod  # necesario para @dataclass en Python 3.10
    )
    spec.loader.exec_module(mod)
    rec = mod.compute_properties(
        "CC1=CC(=NO1)NC(=O)NC2=CC(=C(C=C2OC)OC)Cl", "PNU-120596"  # CID 311434
    )
    # Verificar que el módulo calcula propiedades correctamente
    assert not rec.parse_error, f"Error de parseo: {rec.parse_error}"
    assert rec.MW > 300, f"MW inesperado: {rec.MW}"
    assert rec.CNS_MPO > 0, f"CNS MPO inesperado: {rec.CNS_MPO}"
    return f"MW={rec.MW:.1f}  clogP={rec.clogP:.2f}  CNS_MPO={rec.CNS_MPO}"


check("mol_properties_rdkit.py (funcional)", test_module)


# Imprimir resultados en terminal
print()
print("=" * 60)
print("  Verificación del entorno tfm_alpha7")
print("=" * 60)

n_ok = sum(1 for ok, _, _ in results if ok)
n_fail = sum(1 for ok, _, _ in results if not ok)

for ok, label, version in results:
    symbol = "✓" if ok else "✗"
    tag = "" if ok else "  ← FALTA o ERROR"
    print(f"  {symbol} {label:<40} {version}{tag}")

print()
print(f"  {n_ok}/{len(results)} componentes OK", end="")
if n_fail == 0:
    print("  — Entorno listo para ejecutar el pipeline. ✓")
else:
    print(f"  — {n_fail} componente(s) con error. Revisa los mensajes anteriores.")
print("=" * 60)

# Código de salida
sys.exit(0 if n_fail == 0 else 1)
