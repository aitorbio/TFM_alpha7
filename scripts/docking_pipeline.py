"""
docking_pipeline.py  — Módulos 3, 4 y 5 del pipeline SBDD α7 nAChR
====================================================================
Implementa tres etapas secuenciales del protocolo de cribado virtual
centrado en el sitio alostérico TMD intrasubunitario del receptor
nicotínico de acetilcolina α7 (nAChR α7):

  Módulo 3a — Validación por redocking (§3.5.1):
      Redocking del ligando co-cristalizado (PNU-120596, PDB 8V82) con
      10 ejecuciones independientes. El protocolo se considera validado
      si el RMSD < 2.0 Å en ≥ 7 de 10 ejecuciones.

  Módulo 3b — Benchmarking de enriquecimiento (§3.5.2):
      Docking de activos conocidos (PAMs) y decoys property-matched
      generados desde la biblioteca ZINC-22. Se calculan AUC-ROC,
      EF1%, EF5% y BEDROC para cuantificar la capacidad discriminativa
      del protocolo antes del cribado prospectivo.

  Módulo 4  — Cribado virtual prospectivo (§3.6):
      Conversión de la biblioteca filtrada a PDBQT (OpenBabel + MMFF94),
      docking paralelo con AutoDock Vina 1.2.0 y recopilación de scores.

  Módulo 5  — Análisis IFP y puntuación compuesta (§3.7):
      Cálculo de la huella digital de interacción residuo-ligando con
      ProLIF, comparación con el IFP de referencia (PNU-120596) mediante
      similitud de Jaccard, y cálculo de la puntuación compuesta
      (0.5·S_dock + 0.3·S_IFP + 0.2·S_CNS).

NOTA: El análisis de selectividad TMD vs ECD (Módulo 6) se realiza con
el script independiente docking_ecd_selectivity.py, que emplea un grid
centrado en el sitio ortostérico del ECD (centroide de epibatidina en
PDB 8V82). Reutilizar el grid TMD de este script para ese análisis sería
metodológicamente incorrecto: ambos sitios están separados ~45 Å.

Dependencias externas (deben estar en PATH):
    AutoDock Vina 1.2.0    vina --version
    OpenBabel 3.1           obabel --version
    fpocket 4.0             fpocket --help

Dependencias Python:
    rdkit >= 2023.09
    prolif >= 1.1
    mdanalysis
    pandas, numpy, scipy, scikit-learn, matplotlib

Uso:
    # Preparar receptor + benchmarking + cribado completo
    python docking_pipeline.py \\
        --receptor  data/8V82_prepared.pdbqt \\
        --library   results/library_filtered.csv \\
        --actives   data/pam2_actives.smi \\
        --ref-ligand data/PNU120596.pdbqt \\
        --output    results/ --figures figures/

    # Solo validación (redocking + benchmarking)
    python docking_pipeline.py --validate-only \\
        --receptor data/8V82_prepared.pdbqt \\
        --ref-ligand data/PNU120596.pdbqt \\
        --actives data/pam2_actives.smi

Notas sobre preparación del receptor (paso previo manual):
    1. Descargar PDB 8V82: wget https://files.rcsb.org/download/8V82.pdb
    2. Eliminar agua/ligandos: grep "^ATOM" 8V82.pdb > 8V82_protein.pdb
    3. Añadir H con H++ (web) o reduce: reduce 8V82_protein.pdb > 8V82_H.pdb
    4. Convertir a PDBQT: obabel 8V82_H.pdb -O 8V82_prepared.pdbqt -xr
    5. Extraer ligando ref: obabel 8V82.pdb -O PNU120596.pdbqt --resname UNL
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Verificación de herramientas externas en PATH
# Se comprueba la disponibilidad de Vina, OpenBabel y fpocket antes de iniciar
# el pipeline para evitar errores a mitad de un cribado largo.
# ─────────────────────────────────────────────────────────────────────────────


def _check_tool(name: str, test_arg: str = "--version") -> bool:
    """Verifica que una herramienta externa está en PATH."""
    try:
        subprocess.run([name, test_arg], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_dependencies() -> dict[str, bool]:
    tools = {
        "vina": _check_tool("vina"),
        "obabel": _check_tool("obabel"),
        "fpocket": _check_tool("fpocket", "--help"),
    }
    for name, ok in tools.items():
        level = logging.INFO if ok else logging.WARNING
        log.log(level, f"  {'✓' if ok else '✗'} {name}")
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# Configuración del grid de docking para el sitio TMD intrasubunitario (§3.3.2)
# Las dimensiones (24 × 24 × 24 Å) se establecieron para cubrir el bolsillo
# hidrofóbico del sitio TMD sin solapar con el canal iónico central ni con
# el dominio extracelular (ECD). La exhaustividad de 32 reproduce los
# parámetros reportados en Bouchouireb et al. (2024) para este sitio.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GridConfig:
    """Parámetros del grid de docking para el sitio TMD intrasubunitario."""

    center_x: float = 0.0  # Se recalcula desde fpocket o ligando co-cristalizado
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 24.0  # Å — §3.3.2 metodología
    size_y: float = 24.0
    size_z: float = 24.0
    exhaustiveness: int = 32  # §3.5 metodología, Bouchouireb et al. 2024
    n_poses: int = 20


def get_grid_from_ligand(ligand_pdbqt: str) -> GridConfig:
    """
    Calcula el centro del grid como centroide del ligando co-cristalizado.
    Método más preciso para el redocking y el cribado prospectivo.
    """
    coords = []
    with open(ligand_pdbqt) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    if not coords:
        raise ValueError(f"No se encontraron coordenadas en {ligand_pdbqt}")
    centroid = np.mean(coords, axis=0)
    log.info(f"Centroide del ligando: {centroid}")
    return GridConfig(
        center_x=round(float(centroid[0]), 3),
        center_y=round(float(centroid[1]), 3),
        center_z=round(float(centroid[2]), 3),
    )


def get_grid_from_fpocket(receptor_pdb: str) -> GridConfig:
    """
    Detecta el sitio alostérico TMD con fpocket y devuelve las coordenadas
    del bolsillo más grande compatible con el sitio transmembrana.
    Requiere fpocket en PATH.
    """
    if not _check_tool("fpocket", "--help"):
        raise RuntimeError("fpocket no encontrado en PATH")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["fpocket", "-f", receptor_pdb, "-d", tmp],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"fpocket falló: {result.stderr[:200]}")

        # Leer pocket más grande (pocket1)
        pocket_file = Path(tmp) / (Path(receptor_pdb).stem + "_out") / "pocket1_atm.pdb"
        if not pocket_file.exists():
            raise FileNotFoundError(f"fpocket no generó {pocket_file}")

        coords = []
        with open(pocket_file) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x, y, z = (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
                        coords.append([x, y, z])
                    except ValueError:
                        continue

    if not coords:
        raise ValueError("fpocket no devolvió coordenadas válidas")
    centroid = np.mean(coords, axis=0)
    return GridConfig(
        center_x=round(float(centroid[0]), 3),
        center_y=round(float(centroid[1]), 3),
        center_z=round(float(centroid[2]), 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Preparación de ligandos: SMILES → PDBQT mediante OpenBabel
# El flujo de conversión es: SMILES → 3D (MMFF94) → protonación a pH 7.4
# → hidrógenos explícitos → cargas parciales Gasteiger → PDBQT.
# OpenBabel aplica el campo de fuerza MMFF94 para la generación y
# minimización de la geometría 3D, seguido de la asignación de cargas
# Gasteiger requeridas por AutoDock Vina para el cálculo de interacciones.
# ─────────────────────────────────────────────────────────────────────────────


def smiles_to_pdbqt(smiles: str, name: str, out_path: str, ph: float = 7.4) -> bool:
    """
    Convierte SMILES a PDBQT mediante OpenBabel:
      SMILES → 3D (MMFF94) → protona a pH 7.4 → PDBQT con cargas Gasteiger.

    Devuelve True si la conversión fue exitosa.
    """
    if not _check_tool("obabel"):
        raise RuntimeError("OpenBabel (obabel) no encontrado en PATH")

    with tempfile.NamedTemporaryFile(suffix=".smi", mode="w", delete=False) as f:
        f.write(f"{smiles} {name}\n")
        smi_file = f.name

    try:
        result = subprocess.run(
            [
                "obabel",
                smi_file,
                "-O",
                out_path,
                "--gen3d",  # generar geometría 3D
                "-p",
                str(ph),  # protonación a pH 7.4
                "--minimize",  # minimización MMFF94
                "--ff",
                "MMFF94",
                "-h",  # añadir H explícitos
                "--title",
                name,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        os.unlink(smi_file)

    if result.returncode != 0 or not Path(out_path).exists():
        log.debug(f"obabel falló para {name}: {result.stderr[:100]}")
        return False
    return True


def batch_smiles_to_pdbqt(
    df: pd.DataFrame,
    out_dir: str,
    smiles_col: str = "smiles_canon",
    name_col: str = "name",
) -> list[tuple[str, str]]:
    """
    Convierte toda la biblioteca filtrada a PDBQT.
    Devuelve lista de (name, pdbqt_path) para los que tuvieron éxito.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    n_ok, n_fail = 0, 0

    for _, row in df.iterrows():
        smi = row[smiles_col]
        name = str(row[name_col]).replace("/", "_").replace(" ", "_")
        out = str(Path(out_dir) / f"{name}.pdbqt")

        if smiles_to_pdbqt(smi, name, out):
            results.append((name, out))
            n_ok += 1
        else:
            n_fail += 1

        if (n_ok + n_fail) % 100 == 0:
            log.info(
                f"  Conversión PDBQT: {n_ok+n_fail} procesados "
                f"({n_ok} OK, {n_fail} fallos)"
            )

    log.info(f"Conversión PDBQT: {n_ok} exitosos, {n_fail} fallos")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Docking con AutoDock Vina 1.2.0
# run_vina() ejecuta un único ligando y devuelve el mejor score (kcal/mol).
# run_vina_screening() paraleliza el cribado sobre la biblioteca completa
# usando ProcessPoolExecutor, invocando Vina con --cpu 1 por proceso para
# evitar conflictos cuando múltiples instancias corren simultáneamente.
# ─────────────────────────────────────────────────────────────────────────────


def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    grid: GridConfig,
    log_file: Optional[str] = None,
) -> Optional[float]:
    """
    Ejecuta AutoDock Vina 1.2.0 para un ligando y devuelve el mejor score
    (kcal/mol). Devuelve None si Vina falla.
    """
    cmd = [
        "vina",
        "--receptor",
        receptor_pdbqt,
        "--ligand",
        ligand_pdbqt,
        "--out",
        output_pdbqt,
        "--center_x",
        str(grid.center_x),
        "--center_y",
        str(grid.center_y),
        "--center_z",
        str(grid.center_z),
        "--size_x",
        str(grid.size_x),
        "--size_y",
        str(grid.size_y),
        "--size_z",
        str(grid.size_z),
        "--exhaustiveness",
        str(grid.exhaustiveness),
        "--num_modes",
        str(grid.n_poses),
        "--cpu",
        "1",
    ]
    if log_file:
        cmd += ["--log", log_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning(f"Vina timeout para {ligand_pdbqt}")
        return None

    if result.returncode != 0:
        log.debug(f"Vina error: {result.stderr[:100]}")
        return None

    # Parsear la tabla de resultados de Vina
    # La salida tiene líneas como "   1       -8.234      0.000      0.000"
    for line in result.stdout.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                continue

    # Alternativamente, parsear el PDBQT de salida
    if Path(output_pdbqt).exists():
        with open(output_pdbqt) as f:
            for line in f:
                if "VINA RESULT" in line or "RESULT" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if val < 0:
                                return val
                        except ValueError:
                            continue
    return None


def run_vina_screening(
    receptor_pdbqt: str,
    ligand_pdbqts: list[tuple[str, str]],
    grid: GridConfig,
    out_dir: str,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ejecuta el cribado virtual asíncrono sobre la lista de ligandos PDBQT.
    Devuelve DataFrame con columnas: name, pdbqt_out, dock_score.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    n = len(ligand_pdbqts)

    workers = max_workers or os.cpu_count() or 4
    log.info(f"Cribado paralelo con {workers} procesos...")

    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for name, lig_path in ligand_pdbqts:
            out_path = str(Path(out_dir) / f"{name}_out.pdbqt")
            # En paralelo, `grid.exhaustiveness` debe bajarse o usarse 1 CPU por Vina
            # run_vina ya invoca vina con --cpu 1
            fut = executor.submit(run_vina, receptor_pdbqt, lig_path, out_path, grid)
            futures[fut] = (name, out_path)

        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            name, out_path = futures[fut]
            score = fut.result()
            results.append(
                {
                    "name": name,
                    "pdbqt_out": out_path if score is not None else "",
                    "dock_score": score if score is not None else np.nan,
                }
            )
            completed += 1
            if completed % 50 == 0:
                n_done = sum(1 for r in results if not np.isnan(r["dock_score"]))
                log.info(f"  Cribado: {completed}/{n} ({n_done} scores válidos)")

    df = pd.DataFrame(results)
    log.info(f"Cribado completado: {df['dock_score'].notna().sum()}/{n} exitosos")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validación del protocolo por redocking y cálculo de RMSD (§3.5.1)
# calc_rmsd_pdbqt() implementa el algoritmo de Kabsch (SVD) para la
# superposición óptima de átomos pesados y el cálculo del RMSD resultante.
# run_redocking_validation() realiza 10 ejecuciones independientes con
# semillas distintas y valida el protocolo si ≥ 7/10 runs dan RMSD < 2.0 Å,
# umbral estándar para docking de alta confianza (Morris et al., 2009).
# ─────────────────────────────────────────────────────────────────────────────


def calc_rmsd_pdbqt(ref_pdbqt: str, pose_pdbqt: str) -> float:
    """
    Calcula el RMSD entre la pose de referencia y la pose de docking.
    Usa mínimos cuadrados sobre todos los átomos pesados.
    """

    def read_coords(path: str) -> np.ndarray:
        coords = []
        with open(path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x, y, z = (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
                        coords.append([x, y, z])
                    except ValueError:
                        continue
        return np.array(coords)

    ref = read_coords(ref_pdbqt)
    pose = read_coords(pose_pdbqt)

    if len(ref) == 0 or len(pose) == 0:
        return np.inf
    if len(ref) != len(pose):
        # Tomar el mínimo número de átomos (puede diferir por H explícitos)
        n = min(len(ref), len(pose))
        ref, pose = ref[:n], pose[:n]

    # Superposición por mínimos cuadrados (Algoritmo de Kabsch completo)
    ref_c = ref - ref.mean(axis=0)
    pose_c = pose - pose.mean(axis=0)
    # Matriz de covarianza
    C = np.dot(pose_c.T, ref_c)
    # SVD
    V, S, W = np.linalg.svd(C)
    # Matriz de rotación
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    # Rotar pose
    pose_rotated = np.dot(pose_c, U)
    # Calcular RMSD
    return float(np.sqrt(np.mean(np.sum((ref_c - pose_rotated) ** 2, axis=1))))


def run_redocking_validation(
    receptor_pdbqt: str,
    ref_ligand_pdbqt: str,
    grid: GridConfig,
    n_runs: int = 10,
    rmsd_threshold: float = 2.0,
) -> dict:
    """
    Redocking del ligando co-cristalizado (§3.5.1).
    Ejecuta n_runs ejecuciones independientes con semillas distintas.
    Criterio: RMSD < rmsd_threshold en ≥ 7/10 runs.
    """
    if not _check_tool("vina"):
        raise RuntimeError("AutoDock Vina no encontrado en PATH")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_runs):
            out_path = str(Path(tmp) / f"redock_run{i+1}.pdbqt")
            score = run_vina(receptor_pdbqt, ref_ligand_pdbqt, out_path, grid)
            if score is None or not Path(out_path).exists():
                results.append(
                    {"run": i + 1, "score": np.nan, "rmsd": np.nan, "pass": False}
                )
                continue
            # Extraer solo la mejor pose del PDBQT multi-modelo
            best_pose_path = str(Path(tmp) / f"best_pose_{i+1}.pdbqt")
            _extract_best_pose(out_path, best_pose_path)
            rmsd = calc_rmsd_pdbqt(ref_ligand_pdbqt, best_pose_path)
            passed = bool(rmsd < rmsd_threshold)
            results.append(
                {
                    "run": i + 1,
                    "score": float(score),
                    "rmsd": float(rmsd),
                    "pass": passed,
                }
            )
            log.info(
                f"  Run {i+1:>2}: score={score:.3f}  RMSD={rmsd:.3f} Å  "
                f"{'✓' if passed else '✗'}"
            )

    df_res = pd.DataFrame(results)
    n_pass = int(df_res["pass"].sum())
    validated = bool(n_pass >= 7)

    return {
        "n_runs": int(n_runs),
        "n_pass": n_pass,
        "validated": validated,
        "rmsd_min": float(df_res["rmsd"].min()),
        "rmsd_mean": float(df_res["rmsd"].mean()),
        "score_best": float(df_res["score"].min()),
        "threshold_A": rmsd_threshold,
        "per_run": df_res.to_dict(orient="records"),
    }


def _extract_best_pose(multi_pdbqt: str, out_path: str) -> None:
    """Extrae el primer modelo (mejor pose) de un PDBQT multi-modelo de Vina."""
    lines = []
    with open(multi_pdbqt) as f:
        for line in f:
            if line.startswith("MODEL") and lines:
                break  # second model — stop
            lines.append(line)
    with open(out_path, "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking de enriquecimiento (§3.5.2)
# Se calculan métricas de enriquecimiento estándar en cribado virtual:
#   AUC-ROC : área bajo la curva ROC (umbral ≥ 0.70)
#   EF1%    : factor de enriquecimiento al 1% de la base de datos (umbral ≥ 5.0)
#   EF5%    : factor de enriquecimiento al 5% (umbral ≥ 3.0)
#   BEDROC  : métrica ponderada exponencialmente para enriquecimiento
#             temprano (Truchon & Bayly, 2007; umbral ≥ 0.50)
# Los decoys se generan por property-matching (MW ±50 Da, logP ±0.5,
# TPSA ±20 Å²) con restricción de disimilitud Tanimoto < 0.35 respecto
# a los activos, en línea con la metodología DUD-E (Mysinger et al., 2012).
# ─────────────────────────────────────────────────────────────────────────────


def _bedroc(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, J Chem Inf Model 2007)."""
    n = len(y_true)
    na = y_true.sum()
    if na == 0 or na == n:
        return 0.0
    idx = np.argsort(-scores)
    ys = y_true[idx]
    Ra = na / n
    ri = sum(np.exp(-alpha * i / n) for i, v in enumerate(ys) if v == 1)
    nf = (
        Ra
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra) + 1e-12)
    )
    ri_max = (1 - np.exp(-alpha * Ra)) / (1 - np.exp(-alpha) + 1e-12)
    ri_min = (1 - np.exp(alpha * Ra)) / (1 - np.exp(alpha) + 1e-12)
    bv = (ri / n) * (alpha / (1 - np.exp(-alpha) + 1e-12)) / (nf + 1e-12)
    return float(np.clip((bv - ri_min) / (ri_max - ri_min + 1e-12), 0, 1))


def _ef(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.01) -> float:
    """Enrichment Factor al frac% de la base de datos."""
    n = len(y_true)
    na = y_true.sum()
    ns = max(1, int(n * frac))
    idx = np.argsort(-scores)
    found = y_true[idx[:ns]].sum()
    return float(found / (na * frac)) if na * frac > 0 else 0.0


def generate_property_matched_decoys(
    actives: list[tuple[str, str]],
    background_pool: list[tuple[str, str]],
    decoys_per_active: int = 50,
    max_tanimoto: float = 0.35,
) -> pd.DataFrame:
    """
    Selecciona decoys desde un pool inactivo que coinciden en MW, logP y TPSA
    con los activos, pero que tienen topología disimilar (Tanimoto < 0.35).
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
    from rdkit import Chem

    log.info("Calculando propiedades de los activos...")
    actives_props = []
    actives_fps = []

    for smi, name in actives:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            actives_props.append((mw, logp, tpsa, name))
            actives_fps.append(fp)

    log.info(
        f"Buscando {decoys_per_active} decoys por activo en pool de {len(background_pool)} compuestos..."
    )

    decoys_found = set()
    decoy_rows = []
    sample_pool = background_pool[:50000]

    for amw, alogp, atpsa, aname in actives_props:
        count = 0
        for smi, dname in sample_pool:
            if dname in decoys_found:
                continue

            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue

            dmw = Descriptors.ExactMolWt(mol)
            dlogp = Descriptors.MolLogP(mol)
            dtpsa = Descriptors.TPSA(mol)

            if (
                abs(dmw - amw) < 50
                and abs(dlogp - alogp) < 0.5
                and abs(dtpsa - atpsa) < 20
            ):
                dfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sims = DataStructs.BulkTanimotoSimilarity(dfp, actives_fps)

                if max(sims) < max_tanimoto:
                    decoys_found.add(dname)
                    decoy_rows.append(
                        {"smiles_canon": smi, "name": dname, "decoy_for": aname}
                    )
                    count += 1
                    if count >= decoys_per_active:
                        break
        log.info(f"  Activo {aname}: encontrados {count} decoys adecuados.")

    return pd.DataFrame(decoy_rows)


def run_benchmarking(
    receptor_pdbqt: str,
    actives_smi: str,  # fichero .smi con PAMs activos conocidos
    grid: GridConfig,
    work_dir: str,
    library_csv: Optional[str] = None,
) -> dict:
    """
    Benchmarking de enriquecimiento completo (§3.5.2).
    1. Convierte activos + decoys DUD-E a PDBQT.
    2. Dockea todos con Vina.
    3. Calcula AUC-ROC, EF1%, EF5%, BEDROC.

    Nota: la generación de decoys DUD-E requiere acceso al servidor
    http://dude.docking.org/generate — si no hay red, usar decoys pre-generados
    pasando un fichero decoys.smi junto a activos.smi.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    pdbqt_dir = str(Path(work_dir) / "pdbqt_bench")

    # Leer activos
    from mol_properties_rdkit import read_input

    actives = read_input(actives_smi)
    log.info(f"Benchmarking: {len(actives)} activos")

    # Convertir a PDBQT
    actives_pdbqt = batch_smiles_to_pdbqt(
        pd.DataFrame(actives, columns=["smiles_canon", "name"]),
        pdbqt_dir,
    )

    decoys_file = str(Path(actives_smi).parent / "decoys.smi")
    if Path(decoys_file).exists():
        decoys = read_input(decoys_file)
        log.info(f"  Decoys pregenerados cargados: {len(decoys)}")
        decoys_pdbqt = batch_smiles_to_pdbqt(
            pd.DataFrame(decoys, columns=["smiles_canon", "name"]),
            pdbqt_dir,
        )
    elif library_csv and Path(library_csv).exists():
        log.info(f"Generando decoys desde biblioteca {library_csv}")
        df_lib = pd.read_csv(library_csv)
        if "smiles_canon" in df_lib.columns and "name" in df_lib.columns:
            pool = list(zip(df_lib["smiles_canon"], df_lib["name"]))
            df_decoys = generate_property_matched_decoys(actives, pool)
            decoys_pdbqt = batch_smiles_to_pdbqt(df_decoys, pdbqt_dir)
        else:
            log.warning("Formato de biblioteca no compatible para generar decoys")
            decoys_pdbqt = []
    else:
        log.warning(
            "No se encontró decoys.smi ni library — benchmarking solo con activos"
        )
        decoys_pdbqt = []

    all_pdbqt = actives_pdbqt + decoys_pdbqt
    if not all_pdbqt:
        raise RuntimeError("No hay ligandos válidos para benchmarking")

    # Docking
    df_bench = run_vina_screening(receptor_pdbqt, all_pdbqt, grid, work_dir)
    df_bench = df_bench.dropna(subset=["dock_score"])

    n_act = len(actives_pdbqt)
    names_active = {name for name, _ in actives_pdbqt}
    y_true = np.array(
        [1 if r["name"] in names_active else 0 for _, r in df_bench.iterrows()]
    )
    neg_sc = -df_bench["dock_score"].values  # negado: mayor = mejor activo

    fpr, tpr, _ = roc_curve(y_true, neg_sc)
    roc_auc = auc(fpr, tpr)
    ef1 = _ef(y_true, neg_sc, 0.01)
    ef5 = _ef(y_true, neg_sc, 0.05)
    bd = _bedroc(y_true, neg_sc, 20.0)

    metrics = {
        "AUC_ROC": round(roc_auc, 4),
        "EF1pct": round(ef1, 2),
        "EF5pct": round(ef5, 2),
        "BEDROC": round(bd, 4),
        "n_actives": n_act,
        "n_decoys": len(decoys_pdbqt),
        "validated": roc_auc >= 0.70 and ef1 >= 5.0 and bd >= 0.50,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "y_true": y_true.tolist(),
        "scores": neg_sc.tolist(),
    }
    log.info(
        f"  AUC-ROC={roc_auc:.4f}  EF1%={ef1:.2f}  BEDROC={bd:.4f}  "
        f"→ {'VALIDADO ✓' if metrics['validated'] else 'NO VALIDADO ✗'}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Huella digital de interacción residuo-ligando con ProLIF (§3.7.1)
# calc_ifp_prolif() utiliza ProLIF y MDAnalysis para identificar contactos
# específicos (enlaces de hidrógeno, interacciones hidrofóbicas, π-stacking,
# etc.) entre la pose de docking y los residuos del sitio TMD.
# El IFP se representa como un conjunto de etiquetas "Residuo-TipoInteracción"
# y la similitud con el IFP de referencia (PNU-120596) se calcula mediante
# el coeficiente de Jaccard (intersección/unión).
# Solo se analiza la cadena A del receptor pentamérico para evitar
# asignaciones erróneas de contactos a subunidades equivalentes (cadenas B–E).
# ─────────────────────────────────────────────────────────────────────────────


def calc_ifp_prolif(
    receptor_pdb: str,
    pose_pdbqt: str,
    ref_ifp: Optional[np.ndarray] = None,
    chain: str = "A",
) -> tuple[np.ndarray, float]:
    """
    Calcula la huella digital de interacción residuo-ligando con ProLIF.

    Requiere: pip install prolif mdanalysis

    Parámetros:
        receptor_pdb : receptor en formato PDB (con H explícitos)
        pose_pdbqt   : pose de docking del ligando (mejor pose PDBQT)
        ref_ifp      : vector IFP de referencia para calcular similitud Tanimoto

    Devuelve: (ifp_vector, tanimoto_vs_reference)
    """
    try:
        import MDAnalysis as mda
        import prolif as plf
        from rdkit.DataStructs import TanimotoSimilarity
        from rdkit import DataStructs
    except ImportError:
        raise ImportError(
            "ProLIF/MDAnalysis no encontrado.\n"
            "Instala con: pip install prolif mdanalysis"
        )

    # Convertir PDBQT a PDB para ProLIF
    # Convertir PDBQT a PDB para ProLIF
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_lig:
        lig_pdb = f_lig.name

    # Crear una versión limpia del receptor:
    #   - Sin líneas CONECT (causan errores en RDKit)
    #   - Solo cadena A (la que contiene el sitio TMD de PNU-120596)
    #     El α7 nAChR es un pentámero (cadenas A–E); si se incluyen todas,
    #     ProLIF asigna contactos a residuos equivalentes de otras subunidades
    #     con numeración idéntica pero cadena distinta, dando Tanimoto=0.
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f_prot:
        prot_clean_pdb = f_prot.name
        with open(receptor_pdb, "r") as fin, open(prot_clean_pdb, "w") as fout:
            for line in fin:
                if line.startswith("CONECT"):
                    continue  # eliminar CONECT
                if line.startswith("ATOM") and line[21] != chain:
                    continue  # conservar SOLO la cadena del sitio
                fout.write(line)

    subprocess.run(["obabel", pose_pdbqt, "-O", lig_pdb], capture_output=True)

    try:
        from rdkit import Chem

        # Cargar ligando con RDKit (mejor que MDA para PDBs de obabel)
        mol_lig = Chem.MolFromPDBFile(lig_pdb, sanitize=True, removeHs=False)
        if mol_lig is None:
            mol_lig = Chem.MolFromPDBFile(lig_pdb, sanitize=False, removeHs=False)
        lig = plf.Molecule.from_rdkit(mol_lig)

        # Cargar receptor con RDKit
        mol_prot = Chem.MolFromPDBFile(prot_clean_pdb, sanitize=False, removeHs=False)
        if mol_prot is None:
            raise ValueError("Error leyendo PDB")

        try:
            mol_prot.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol_prot,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            pass

        prot = plf.Molecule.from_rdkit(mol_prot)
        fp = plf.Fingerprint()
        fp.run_from_iterable([lig], prot)

        ifp_set = set()
        if not fp.to_dataframe().empty:
            for col in fp.to_dataframe().columns:
                res_full = str(col[1])
                res_name = res_full.split(".")[0]
                int_type = str(col[2])
                ifp_set.add(f"{res_name}-{int_type}")

        # LOG DE DIAGNÓSTICO (Esto nos dirá la verdad)
        if ref_ifp is not None:
            log.info(
                f"      -> {Path(pose_pdbqt).stem}: {len(ifp_set)} interacciones detectadas"
            )

    finally:
        if os.path.exists(lig_pdb):
            os.unlink(lig_pdb)
        if os.path.exists(prot_clean_pdb):
            os.unlink(prot_clean_pdb)

    if not ifp_set:
        # log.debug("      0 interacciones detectadas para esta pose.")
        return np.array([]), 0.0

    # Calcular similitud Tanimoto basada en conjuntos (Sets)
    sim = 0.0
    if ref_ifp is not None and isinstance(ref_ifp, set):
        intersection = ifp_set.intersection(ref_ifp)
        union = ifp_set.union(ref_ifp)
        sim = float(len(intersection) / len(union)) if len(union) > 0 else 0.0

        # Si queremos ver qué está pasando (solo depuración)
        # if len(intersection) > 0:
        #    log.info(f"      Coincidencias: {len(intersection)}")

    return ifp_set, sim


# ─────────────────────────────────────────────────────────────────────────────
# Puntuación compuesta multi-criterio (§3.6.2)
# Integra el score de docking (normalizado por min-max), la similitud IFP
# y el CNS MPO en una puntuación única para la priorización de candidatos.
# Las ponderaciones (w1=0.5, w2=0.3, w3=0.2) son decisiones a priori
# que priorizan la complementariedad energética sobre la semejanza de
# contactos residuales y las propiedades CNS (ver §3.6.2 para justificación
# y análisis de sensibilidad).
# ─────────────────────────────────────────────────────────────────────────────


def composite_score(
    dock_scores: np.ndarray,
    ifp_sims: np.ndarray,
    cns_mpo: np.ndarray,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> np.ndarray:
    """
    Puntuación compuesta = w1·S_dock + w2·S_IFP + w3·S_CNS.
    S_dock se normaliza con min-max (más negativo → más cercano a 1).
    Las ponderaciones son decisiones a priori (ver §3.6.2 limitaciones).
    """
    # Manejar NaNs en el score de docking (asignar 0 si es NaN)
    valid = ~np.isnan(dock_scores)
    s_dock = np.zeros_like(dock_scores)
    if valid.sum() > 1:
        dmin, dmax = np.nanmin(dock_scores), np.nanmax(dock_scores)
        if abs(dmin - dmax) > 1e-9:
            s_dock[valid] = np.clip((dock_scores[valid] - dmax) / (dmin - dmax), 0, 1)

    # Manejar NaNs en IFP y MPO (rellenar con 0)
    s_ifp = np.nan_to_num(ifp_sims, nan=0.0)
    s_mpo = np.nan_to_num(cns_mpo, nan=0.0) / 6.0

    return w1 * s_dock + w2 * np.clip(s_ifp, 0, 1) + w3 * np.clip(s_mpo, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Figura de benchmarking
# ─────────────────────────────────────────────────────────────────────────────


def plot_benchmarking(bench: dict, out_path: str) -> None:
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        f"Módulo 3 — Validación del protocolo de docking\n"
        f"AUC-ROC={bench['AUC_ROC']:.3f}  |  EF1%={bench['EF1pct']:.1f}  |  "
        f"BEDROC={bench['BEDROC']:.3f}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    fpr = np.array(bench["fpr"])
    tpr = np.array(bench["tpr"])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, "#1976D2", lw=2.5, label=f"AUC = {bench['AUC_ROC']:.3f}")
    ax1.fill_between(fpr, tpr, alpha=0.12, color="#1976D2")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Azar")
    ax1.axhline(0.70, color="#2E7D32", ls="--", lw=1.5, label="Criterio ≥ 0.70")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("Curva ROC", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    n = len(bench["y_true"])
    scores = np.array(bench["scores"])
    yt = np.array(bench["y_true"])
    na = int(yt.sum())
    idx = np.argsort(-scores)
    cum = np.cumsum(yt[idx]) / max(na, 1)
    frac = np.arange(1, n + 1) / n
    ax2.plot(frac * 100, cum * 100, "#E91E63", lw=2.5, label="Protocolo")
    ax2.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Aleatorio")
    ef1y = float(np.interp(0.01, frac, cum)) * 100
    ax2.annotate(
        f"EF1%={bench['EF1pct']:.1f}",
        xy=(1, ef1y),
        xytext=(8, ef1y - 10),
        arrowprops=dict(arrowstyle="->", color="#FF5722"),
        fontsize=8,
        color="#FF5722",
    )
    ax2.set_xlabel("% base de datos")
    ax2.set_ylabel("% activos")
    ax2.set_title("Curva de Enriquecimiento", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    scores_raw = -np.array(bench["scores"])  # volver a escala original
    ax3.hist(
        scores_raw[:na],
        bins=15,
        alpha=0.75,
        color="#43A047",
        label=f"Activos (n={na})",
        density=True,
        edgecolor="white",
    )
    ax3.hist(
        scores_raw[na:],
        bins=25,
        alpha=0.6,
        color="#EF5350",
        label=f"Decoys (n={n-na})",
        density=True,
        edgecolor="white",
    )
    ax3.set_xlabel("Docking score (kcal/mol)")
    ax3.set_ylabel("Densidad")
    ax3.set_title("Distribución scores", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    metrics = [
        ("AUC-ROC", bench["AUC_ROC"], 0.70, "≥ 0.70"),
        ("EF 1%", bench["EF1pct"], 5.0, "≥ 5.0"),
        ("EF 5%", bench["EF5pct"], 3.0, "≥ 3.0"),
        ("BEDROC", bench["BEDROC"], 0.50, "≥ 0.50"),
    ]
    ax4.text(
        0.5,
        0.96,
        "Resumen Validación",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax4.transAxes,
    )
    y = 0.78
    for nm, val, thr, lbl in metrics:
        ok = val >= thr
        col = "#2E7D32" if ok else "#C62828"
        ax4.add_patch(
            plt.Rectangle(
                (0.05, y - 0.06),
                0.90,
                0.10,
                color=col,
                alpha=0.1,
                transform=ax4.transAxes,
            )
        )
        ax4.text(
            0.10,
            y - 0.005,
            f"{'✓' if ok else '✗'} {nm}",
            ha="left",
            fontsize=11,
            color=col,
            fontweight="bold",
            transform=ax4.transAxes,
        )
        ax4.text(
            0.55,
            y - 0.005,
            f"{val:.3f} ({lbl})",
            ha="left",
            fontsize=9,
            color="#444",
            transform=ax4.transAxes,
        )
        y -= 0.16
    status = bench["validated"]
    sc = "#2E7D32" if status else "#C62828"
    ax4.add_patch(
        plt.Rectangle(
            (0.05, 0.03), 0.90, 0.14, color=sc, alpha=0.15, transform=ax4.transAxes
        )
    )
    ax4.text(
        0.5,
        0.10,
        "✓ PROTOCOLO VALIDADO" if status else "✗ PROTOCOLO NO VALIDADO",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=sc,
        transform=ax4.transAxes,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Módulo 3-5 — Docking con AutoDock Vina + ProLIF"
    )
    ap.add_argument(
        "--receptor",
        "-r",
        required=True,
        help="Receptor PDBQT (protonado, sin agua/ligandos)",
    )
    ap.add_argument(
        "--ref-ligand",
        required=True,
        help="Ligando co-cristalizado PDBQT (PNU-120596 de 8V82)",
    )
    ap.add_argument(
        "--library", "-l", help="Biblioteca filtrada CSV (salida de library_builder.py)"
    )
    ap.add_argument(
        "--actives", "-a", help="Fichero .smi con PAMs activos para benchmarking"
    )
    ap.add_argument("--receptor-pdb", help="Receptor PDB con H (para ProLIF IFP)")
    ap.add_argument("--output", "-o", default="results", help="Directorio de salida")
    ap.add_argument("--figures", default="figures")
    # NOTA: El análisis de selectividad ECD se realiza con el script
    # independiente docking_ecd_selectivity.py, que usa un grid específico
    # para el sitio ortostérico del ECD (centroide de epibatidina).
    ap.add_argument(
        "--workers", type=int, default=None, help="Cores para Vina (default: todos)"
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Solo ejecutar redocking + benchmarking",
    )
    ap.add_argument("--skip-redock", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--exhaustiveness", type=int, default=32)
    ap.add_argument(
        "--ifp-min",
        type=float,
        default=0.30,
        help="Umbral mínimo de similitud IFP (default: 0.30)",
    )
    return ap.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tools = check_dependencies()
    if not tools["vina"]:
        sys.exit(
            "AutoDock Vina no encontrado. Instala desde https://github.com/ccsb-scripps/AutoDock-Vina"
        )
    if not tools["obabel"]:
        sys.exit("OpenBabel no encontrado. conda install -c conda-forge openbabel")

    # Grid desde ligando co-cristalizado
    log.info("Definiendo grid desde ligando co-cristalizado...")
    grid = get_grid_from_ligand(args.ref_ligand)
    grid = GridConfig(
        center_x=grid.center_x,
        center_y=grid.center_y,
        center_z=grid.center_z,
        exhaustiveness=args.exhaustiveness,
    )
    log.info(
        f"  Grid: center=({grid.center_x:.2f}, {grid.center_y:.2f}, "
        f"{grid.center_z:.2f})  size={grid.size_x}×{grid.size_y}×{grid.size_z} Å"
    )

    # ── Módulo 3a: Redocking ─────────────────────────────────────────────────
    if not args.skip_redock:
        log.info("\n── Redocking del ligando co-cristalizado (§3.5.1) ──")
        redock = run_redocking_validation(
            args.receptor, args.ref_ligand, grid, n_runs=10
        )
        (out_dir / "redocking_results.json").write_text(json.dumps(redock, indent=2))
        log.info(
            f"  Redocking: RMSD_min={redock['rmsd_min']:.3f} Å  "
            f"{redock['n_pass']}/10 runs < 2.0 Å  "
            f"→ {'VALIDADO ✓' if redock['validated'] else 'NO VALIDADO ✗'}"
        )

    # ── Módulo 3b: Benchmarking ──────────────────────────────────────────────
    bench_metrics = None
    if not args.skip_bench and args.actives:
        log.info("\n── Benchmarking de enriquecimiento (§3.5.2) ──")
        bench_metrics = run_benchmarking(
            args.receptor,
            args.actives,
            grid,
            str(out_dir / "benchmarking_work"),
            library_csv=args.library,
        )
        # Guardar sin arrays numpy grandes
        bench_save = {
            k: v
            for k, v in bench_metrics.items()
            if k not in ("fpr", "tpr", "y_true", "scores")
        }
        (out_dir / "benchmarking_metrics.json").write_text(
            json.dumps(bench_save, indent=2)
        )
        plot_benchmarking(bench_metrics, str(fig_dir / "fig_benchmarking.png"))

        if not bench_metrics["validated"]:
            log.warning("Protocolo NO validado. Ajusta parámetros antes del cribado.")
            if not args.validate_only:
                log.warning(
                    "Continuando igualmente (--validate-only para detener aquí)."
                )

    if args.validate_only:
        log.info("✓ Validación completada (--validate-only). Fin.")
        return

    if not args.library:
        sys.exit("Error: --library requerido para el cribado")

    # ── Módulo 4: Cribado virtual ────────────────────────────────────────────
    log.info("\n── Módulo 4: Cribado virtual prospectivo (§3.6) ──")
    df_lib = pd.read_csv(args.library)
    log.info(f"  Biblioteca: {len(df_lib)} compuestos")

    pdbqt_dir = str(out_dir / "pdbqt_library")
    log.info("  Convirtiendo SMILES → PDBQT (OpenBabel)...")
    ligand_pdbqts = batch_smiles_to_pdbqt(df_lib, pdbqt_dir)
    log.info(f"  {len(ligand_pdbqts)} ligandos preparados para docking")

    df_docking = run_vina_screening(
        args.receptor,
        ligand_pdbqts,
        grid,
        str(out_dir / "docking_poses"),
        max_workers=args.workers,
    )

    # NOTA: El análisis de selectividad TMD vs ECD se realiza con el script
    # independiente docking_ecd_selectivity.py, que emplea un grid centrado
    # en el sitio ortostérico del ECD (epibatidina) en lugar de reutilizar
    # el grid TMD, lo cual sería metodológicamente incorrecto dado que ambos
    # sitios están separados ~45 Å.

    # Garantizar que la columna "name" es string y está limpia en ambos DataFrames
    df_lib["name"] = df_lib["name"].astype(str).str.strip()
    df_docking["name"] = df_docking["name"].astype(str).str.strip()

    # IMPORTANTE: Si la biblioteca ya tiene columnas de docking (por ser un re-run o un test),
    # las eliminamos para evitar columnas duplicadas tipo "dock_score_x"
    cols_to_drop = [
        c
        for c in ["dock_score", "ifp_similarity", "composite_score", "pdbqt_out"]
        if c in df_lib.columns
    ]
    if cols_to_drop:
        df_lib = df_lib.drop(columns=cols_to_drop)

    # Realizar el merge (ahora sin colisiones de nombres)
    df_results = df_lib.merge(df_docking, on="name", how="inner")

    if df_results.empty:
        log.error(
            "¡Error crítico! No hay coincidencias entre la biblioteca y los resultados de docking."
        )
        log.error(f"Ejemplos lib: {df_lib['name'].head().tolist()}")
        log.error(f"Ejemplos dock: {df_docking['name'].head().tolist()}")
        return

    # ── Módulo 5: IFP + puntuación compuesta ─────────────────────────────────
    log.info("\n── Módulo 5: Análisis IFP y puntuación compuesta (§3.7) ──")

    # IFP de referencia desde ligando co-cristalizado (si hay receptor PDB)
    ref_ifp = None
    if args.receptor_pdb and Path(args.ref_ligand).exists():
        try:
            ref_ifp, _ = calc_ifp_prolif(
                args.receptor_pdb, args.ref_ligand, chain=getattr(args, "chain", "A")
            )
            log.info(f"  IFP de referencia calculado: {len(ref_ifp)} contactos")
            log.info(f"  Interacciones clave: {', '.join(sorted(list(ref_ifp)))}")
        except Exception as e:
            log.warning(f"  ProLIF no disponible: {e}. IFP = 0 para todos.")

    # Calcular IFP para las mejores poses
    ifp_sims = []
    for _, row in df_results.iterrows():
        pose_path = row.get("pdbqt_out", "")
        if (
            not pd.isna(row.get("dock_score"))
            and pose_path
            and Path(pose_path).exists()
            and args.receptor_pdb
            and ref_ifp is not None
        ):
            try:
                _, sim = calc_ifp_prolif(
                    args.receptor_pdb,
                    pose_path,
                    ref_ifp,
                    chain=getattr(args, "chain", "A"),
                )
            except Exception:
                sim = 0.0
        else:
            sim = 0.0
        ifp_sims.append(sim)

    df_results["ifp_similarity"] = ifp_sims

    # Puntuación compuesta
    df_results["composite_score"] = composite_score(
        df_results["dock_score"].values,
        df_results["ifp_similarity"].values,
        df_results["CNS_MPO"].values,
    )

    # Aplicar umbral IFP mínimo
    n_before = len(df_results)
    df_results = df_results[df_results["ifp_similarity"] >= args.ifp_min].copy()
    log.info(f"  Eliminados IFP < {args.ifp_min}: {n_before - len(df_results)}")

    # Top 5%
    thr95 = df_results["composite_score"].quantile(0.95)
    df_top = df_results[df_results["composite_score"] >= thr95].sort_values(
        "composite_score", ascending=False
    )

    # Guardar
    df_results.to_csv(out_dir / "screening_all.csv", index=False)
    df_top.to_csv(out_dir / "hits_top5pct.csv", index=False)

    log.info(f"\n✓ Cribado completado.")
    log.info(f"  Top 5% (composite ≥ {thr95:.3f}): {len(df_top)} candidatos")
    log.info(f"  Ficheros en: {out_dir}")

    # Top 15
    log.info("\nTop 15 candidatos:")
    log.info(
        f"{'Rank':<5} {'ID':<20} {'Score':>14} {'IFP':>6} " f"{'MPO':>6} {'Comp':>7}"
    )
    for i, (_, row) in enumerate(df_top.head(15).iterrows()):
        log.info(
            f"{i+1:<5} {row['name']:<20} {row['dock_score']:>14.2f} "
            f"{row['ifp_similarity']:>6.3f} {row['CNS_MPO']:>6.1f} "
            f"{row['composite_score']:>7.3f}"
        )


if __name__ == "__main__":
    main()
