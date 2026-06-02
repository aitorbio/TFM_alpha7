"""
recover_from_docking.py
=======================
Script de recuperación para continuar el pipeline desde los ficheros
PDBQT ya generados por AutoDock Vina, saltando las 12 h de docking.

Qué hace:
  1. Lee la biblioteca filtrada (library_filtered.csv)
  2. Escanea results/docking_poses/ y extrae el score de cada PDBQT
  3. Reconstruye df_docking con los scores reales de Vina
  4. Aplica el merge con la corrección de tipos (fix del error original)
  5. Calcula IFP + puntuación compuesta
  6. Guarda screening_all.csv y hits_top5pct.csv

Uso:
    conda activate tfm_alpha7
python recover_from_docking.py \
    --library   results/library_filtered.csv \
    --poses-dir results/docking_poses \
    --output    results/ \
    --figures   figures/

    # Con IFP real (requiere receptor PDB y ProLIF):
    python recover_from_docking.py \\
        --library     results/library_filtered.csv \\
        --poses-dir   results/docking_poses \\
        --receptor-pdb data/8V82_protein_H.pdb \\
        --ref-ligand   data/PNU120596.pdbqt \\
        --output results/ --figures figures/
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 1: extraer scores de los PDBQT ya generados
# ─────────────────────────────────────────────────────────────────────────────


def extract_score_from_pdbqt(pdbqt_path: str) -> float | None:
    """
    Lee el mejor score de docking de un fichero PDBQT de salida de Vina.

    Vina escribe el score en líneas de la forma:
        REMARK VINA RESULT:   -8.234      0.000      0.000
    o bien (versiones antiguas):
        REMARK      RESULT:      -8.234    0.000    0.000

    Devuelve el score (número negativo) o None si no se encuentra.
    """
    try:
        with open(pdbqt_path) as f:
            for line in f:
                # Formato estándar Vina 1.2
                if "VINA RESULT" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if val < 0:
                                return val
                        except ValueError:
                            continue
                # Formato alternativo
                if line.startswith("REMARK") and "RESULT" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if val < 0:
                                return val
                        except ValueError:
                            continue
    except (OSError, UnicodeDecodeError):
        return None
    return None


def reconstruct_docking_df(poses_dir: str) -> pd.DataFrame:
    """
    Escanea el directorio de poses y reconstruye el DataFrame de docking
    leyendo cada fichero *_out.pdbqt generado por run_vina_screening.

    El nombre del compuesto se infiere del nombre de fichero:
        ZINC100000003_out.pdbqt  →  name = "ZINC100000003"
    """
    poses_path = Path(poses_dir)
    if not poses_path.exists():
        raise FileNotFoundError(
            f"Directorio de poses no encontrado: {poses_dir}\n"
            "Asegúrate de pasar la ruta correcta con --poses-dir"
        )

    pdbqt_files = sorted(poses_path.glob("*_out.pdbqt"))
    if not pdbqt_files:
        raise FileNotFoundError(
            f"No se encontraron ficheros *_out.pdbqt en {poses_dir}"
        )

    log.info(f"Encontrados {len(pdbqt_files)} ficheros PDBQT en {poses_dir}")

    records = []
    n_ok, n_fail = 0, 0

    for pdbqt in pdbqt_files:
        # Nombre: quitar el sufijo _out.pdbqt
        name = pdbqt.stem.replace("_out", "")
        score = extract_score_from_pdbqt(str(pdbqt))

        if score is not None:
            n_ok += 1
        else:
            n_fail += 1

        records.append(
            {
                "name": name,
                "pdbqt_out": str(pdbqt),
                "dock_score": score if score is not None else np.nan,
            }
        )

    df = pd.DataFrame(records)
    log.info(
        f"  Scores extraídos: {n_ok} OK  |  {n_fail} sin score (Vina falló para estos)"
    )
    log.info(f"  Score mínimo (mejor): {df['dock_score'].min():.3f} kcal/mol")
    log.info(f"  Score medio:          {df['dock_score'].mean():.3f} kcal/mol")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2: IFP con ProLIF (opcional)
# ─────────────────────────────────────────────────────────────────────────────


def calc_ifp_tanimoto(receptor_pdb: str, pose_pdbqt: str, ref_ifp: np.ndarray) -> float:
    """
    Calcula la similitud Tanimoto entre el IFP de una pose y el de referencia
    usando ProLIF + MDAnalysis.
    Devuelve 0.0 si ProLIF no está disponible o falla.
    """
    try:
        import MDAnalysis as mda
        import prolif as plf
    except ImportError:
        return 0.0

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        lig_pdb = f.name

    try:
        subprocess.run(
            ["obabel", pose_pdbqt, "-O", lig_pdb], capture_output=True, timeout=15
        )
        if not Path(lig_pdb).exists() or Path(lig_pdb).stat().st_size == 0:
            return 0.0

        u_prot = mda.Universe(receptor_pdb)
        u_lig = mda.Universe(lig_pdb)
        prot = plf.Molecule.from_mda(u_prot)
        lig = plf.Molecule.from_mda(u_lig)
        fp = plf.Fingerprint()
        fp.run_from_iterable([lig], prot)
        df_fp = fp.to_dataframe()

        if df_fp.empty:
            return 0.0

        ifp_vec = df_fp.values.flatten().astype(float)

        # Adaptar longitud al IFP de referencia
        min_len = min(len(ifp_vec), len(ref_ifp))
        v1 = ifp_vec[:min_len]
        v2 = ref_ifp[:min_len]

        inter = np.dot(v1, v2)
        union = v1.sum() + v2.sum() - inter
        return float(inter / union) if union > 0 else 0.0

    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(lig_pdb)
        except OSError:
            pass


def get_ref_ifp(receptor_pdb: str, ref_ligand_pdbqt: str) -> np.ndarray | None:
    """Calcula el IFP del ligando co-cristalizado como referencia."""
    try:
        import MDAnalysis as mda
        import prolif as plf
    except ImportError:
        log.warning(
            "ProLIF no disponible. IFP se fijará a 0 para todos los candidatos."
        )
        return None

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        lig_pdb = f.name

    try:
        subprocess.run(
            ["obabel", ref_ligand_pdbqt, "-O", lig_pdb], capture_output=True, timeout=15
        )
        u_prot = mda.Universe(receptor_pdb)
        u_lig = mda.Universe(lig_pdb)
        prot = plf.Molecule.from_mda(u_prot)
        lig = plf.Molecule.from_mda(u_lig)
        fp = plf.Fingerprint()
        fp.run_from_iterable([lig], prot)
        df_fp = fp.to_dataframe()
        if df_fp.empty:
            return None
        ref = df_fp.values.flatten().astype(float)
        log.info(
            f"  IFP de referencia: {len(ref)} bits, "
            f"{int(ref.sum())} interacciones activas"
        )
        return ref
    except Exception as e:
        log.warning(f"  No se pudo calcular IFP de referencia: {e}")
        return None
    finally:
        try:
            os.unlink(lig_pdb)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3: puntuación compuesta
# ─────────────────────────────────────────────────────────────────────────────


def composite_score(
    dock_scores: np.ndarray,
    ifp_sims: np.ndarray,
    cns_mpo: np.ndarray,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> np.ndarray:
    """S = w1·S_dock_norm + w2·S_IFP + w3·S_CNS"""
    valid = ~np.isnan(dock_scores)
    s_dock = np.zeros_like(dock_scores)
    if valid.sum() > 1:
        dmin = dock_scores[valid].min()
        dmax = dock_scores[valid].max()
        s_dock[valid] = np.clip(
            (dock_scores[valid] - dmax) / (dmin - dmax + 1e-9), 0, 1
        )
    return (
        w1 * s_dock + w2 * np.clip(ifp_sims, 0, 1) + w3 * np.clip(cns_mpo / 6.0, 0, 1)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paso 4: figura de resultados del cribado
# ─────────────────────────────────────────────────────────────────────────────


def plot_screening_results(df: pd.DataFrame, out_path: str) -> None:
    """Figura resumen del cribado virtual."""
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle(
        f"Módulo 4–5 — Cribado virtual prospectivo\n"
        f"n = {len(df)} compuestos cribados  |  "
        f"Recuperado desde PDBQT existentes",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # 1. Distribución de docking scores
    ax1 = fig.add_subplot(gs[0, 0])
    scores = df["dock_score"].dropna()
    thr5 = scores.quantile(0.05)
    ax1.hist(
        scores, bins=30, color="#78909C", alpha=0.7, edgecolor="white", label="Todos"
    )
    ax1.hist(
        scores[scores <= thr5],
        bins=10,
        color="#E91E63",
        alpha=0.9,
        edgecolor="white",
        label=f"Top 5% (n={int((scores <= thr5).sum())})",
    )
    ax1.axvline(thr5, color="#E91E63", ls="--", lw=2)
    ax1.set_xlabel("Docking score (kcal/mol)", fontsize=10)
    ax1.set_ylabel("Frecuencia", fontsize=10)
    ax1.set_title("Distribución de\nDocking Scores", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # 2. Distribución IFP
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        df["ifp_similarity"], bins=20, color="#5C6BC0", alpha=0.75, edgecolor="white"
    )
    ax2.axvline(0.30, color="#FF5722", ls="--", lw=2, label="Umbral mín. (0.30)")
    ax2.axvline(0.40, color="#2E7D32", ls="--", lw=2, label="Umbral prior. (0.40)")
    ax2.set_xlabel("Similitud IFP (Tanimoto)", fontsize=10)
    ax2.set_ylabel("Frecuencia", fontsize=10)
    ax2.set_title("Similitud IFP\nvs. PNU-120596", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # 3. Scatter docking vs IFP
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(
        df["dock_score"],
        df["ifp_similarity"],
        c=df["composite_score"],
        cmap="RdYlGn",
        alpha=0.65,
        s=25,
        vmin=0,
        vmax=1,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax3, label="Puntuación compuesta")
    ax3.axhline(0.40, color="#2E7D32", ls="--", lw=1.5, alpha=0.7)
    ax3.set_xlabel("Docking score (kcal/mol)", fontsize=10)
    ax3.set_ylabel("Similitud IFP", fontsize=10)
    ax3.set_title("Score vs. IFP", fontsize=10, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)

    # 4. Top 15 candidatos
    ax4 = fig.add_subplot(gs[1, :2])
    top15 = df.nlargest(15, "composite_score").reset_index(drop=True)
    colors_b = [
        "#1B5E20" if s >= 0.7 else "#388E3C" if s >= 0.55 else "#81C784"
        for s in top15["composite_score"]
    ]
    ax4.barh(
        range(len(top15)), top15["composite_score"], color=colors_b, edgecolor="white"
    )
    for i, row in top15.iterrows():
        ax4.text(
            top15["composite_score"].max() + 0.01,
            i,
            f" {row['dock_score']:.1f} kcal/mol | "
            f"IFP={row['ifp_similarity']:.2f} | "
            f"MPO={row['CNS_MPO']:.1f}",
            va="center",
            fontsize=7.5,
            color="#333",
        )
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels(
        [f"#{i+1} {r['name']}" for i, r in top15.iterrows()], fontsize=8
    )
    ax4.set_xlabel("Puntuación compuesta", fontsize=10)
    ax4.set_title("Top 15 candidatos", fontsize=11, fontweight="bold")
    ax4.axvline(0.60, color="#E91E63", ls="--", lw=1.5, label="Umbral (0.60)")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.spines[["top", "right"]].set_visible(False)

    # 5. CNS MPO top candidatos
    ax5 = fig.add_subplot(gs[1, 2])
    dft = df.nlargest(max(3, int(len(df) * 0.1)), "composite_score")
    ax5.hist(dft["CNS_MPO"], bins=8, color="#7B1FA2", alpha=0.8, edgecolor="white")
    ax5.axvline(4.0, color="#FF5722", ls="--", lw=2, label="CNS MPO ≥ 4")
    n_pass = (dft["CNS_MPO"] >= 4).sum()
    ax5.text(
        0.98,
        0.97,
        f"{n_pass}/{len(dft)}\ncon MPO ≥ 4",
        ha="right",
        va="top",
        transform=ax5.transAxes,
        fontsize=9,
        color="#7B1FA2",
        fontweight="bold",
    )
    ax5.set_xlabel("CNS MPO score", fontsize=10)
    ax5.set_ylabel("Frecuencia", fontsize=10)
    ax5.set_title("CNS MPO\nTop candidatos", fontsize=10, fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de recuperación completo
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Recuperación del pipeline desde PDBQT existentes"
    )
    ap.add_argument(
        "--library", "-l", required=True, help="results/library_filtered.csv"
    )
    ap.add_argument(
        "--poses-dir",
        "-p",
        required=True,
        help="results/docking_poses  (directorio con *_out.pdbqt)",
    )
    ap.add_argument("--output", "-o", default="results")
    ap.add_argument("--figures", "-f", default="figures")
    ap.add_argument(
        "--receptor-pdb", help="Receptor PDB con H (para ProLIF IFP, opcional)"
    )
    ap.add_argument(
        "--ref-ligand", help="Ligando de referencia PDBQT (para ProLIF IFP, opcional)"
    )
    ap.add_argument(
        "--ifp-min", type=float, default=0.30, help="Umbral mínimo IFP (default: 0.30)"
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 1: reconstruir df_docking desde los PDBQT ───────────────────────
    log.info("=" * 55)
    log.info("RECUPERACIÓN DEL PIPELINE — Leyendo PDBQT existentes")
    log.info("=" * 55)
    log.info(f"\n[1/5] Escaneando {args.poses_dir}...")
    df_docking = reconstruct_docking_df(args.poses_dir)

    n_total = len(df_docking)
    n_scored = df_docking["dock_score"].notna().sum()
    log.info(f"  {n_scored}/{n_total} compuestos con score válido")

    # ── Paso 2: cargar biblioteca y merge con corrección de tipos ────────────
    log.info(f"\n[2/5] Cargando biblioteca: {args.library}")
    df_lib = pd.read_csv(args.library)
    log.info(f"  {len(df_lib)} compuestos en la biblioteca filtrada")

    # FIX: forzar 'name' a string en ambos DataFrames antes del merge
    df_lib["name"] = df_lib["name"].astype(str)
    df_docking["name"] = df_docking["name"].astype(str)

    df_results = df_lib.merge(df_docking, on="name", how="left")
    log.info(f"  Merge completado: {len(df_results)} filas")

    n_matched = df_results["dock_score"].notna().sum()
    n_missing = df_results["dock_score"].isna().sum()
    if n_missing > 0:
        log.warning(
            f"  {n_missing} compuestos sin score "
            f"(Vina falló o PDBQT no generado para estos)"
        )
    log.info(f"  Compuestos con score: {n_matched}/{len(df_results)}")

    # ── Paso 3: IFP (con ProLIF si está disponible) ──────────────────────────
    log.info(f"\n[3/5] Calculando similitud IFP...")
    ref_ifp = None
    if args.receptor_pdb and args.ref_ligand:
        log.info("  Calculando IFP de referencia (PNU-120596)...")
        ref_ifp = get_ref_ifp(args.receptor_pdb, args.ref_ligand)

    if ref_ifp is None:
        log.info("  IFP de referencia no disponible → ifp_similarity = 0.0 para todos")
        log.info("  (Puedes recalcular con --receptor-pdb y --ref-ligand)")
        df_results["ifp_similarity"] = 0.0
    else:
        log.info(f"  Calculando IFP para {n_matched} poses...")
        ifp_sims = []
        for i, (_, row) in enumerate(df_results.iterrows()):
            pose = row.get("pdbqt_out", "")
            if not pd.isna(row.get("dock_score")) and pose and Path(pose).exists():
                sim = calc_ifp_tanimoto(args.receptor_pdb, pose, ref_ifp)
            else:
                sim = 0.0
            ifp_sims.append(sim)
            if (i + 1) % 100 == 0:
                log.info(f"    IFP: {i+1}/{len(df_results)} procesados...")
        df_results["ifp_similarity"] = ifp_sims

    # ── Paso 4: puntuación compuesta + filtros ────────────────────────────────
    log.info(f"\n[4/5] Calculando puntuación compuesta...")
    df_results["composite_score"] = composite_score(
        df_results["dock_score"].values,
        df_results["ifp_similarity"].values,
        df_results["CNS_MPO"].values,
    )

    # Filtro IFP mínimo (solo si hay IFP real)
    n_before = len(df_results)
    if ref_ifp is not None:
        df_results = df_results[df_results["ifp_similarity"] >= args.ifp_min].copy()
        log.info(
            f"  Eliminados por IFP < {args.ifp_min}: " f"{n_before - len(df_results)}"
        )

    # Top 5%
    df_valid = df_results.dropna(subset=["dock_score"])
    thr95 = df_valid["composite_score"].quantile(0.95)
    df_top = df_valid[df_valid["composite_score"] >= thr95].sort_values(
        "composite_score", ascending=False
    )

    # ── Paso 5: guardar y mostrar resultados ──────────────────────────────────
    log.info(f"\n[5/5] Guardando resultados...")
    df_results.to_csv(out_dir / "screening_all.csv", index=False)
    df_top.to_csv(out_dir / "hits_top5pct.csv", index=False)
    log.info(f"  screening_all.csv  →  {len(df_results)} compuestos")
    log.info(
        f"  hits_top5pct.csv   →  {len(df_top)} candidatos "
        f"(composite ≥ {thr95:.3f})"
    )

    plot_screening_results(df_valid, str(fig_dir / "fig_screening_results.png"))

    log.info("\n" + "=" * 55)
    log.info("✓ Recuperación completada.")
    log.info(f"  Top 5% candidatos: {len(df_top)}")
    log.info("\nTop 15 candidatos:")
    log.info(
        f"  {'Rank':<5} {'ID':<20} {'Score':>14} " f"{'IFP':>6} {'MPO':>6} {'Comp':>7}"
    )
    log.info(f"  {'-'*58}")
    for i, (_, row) in enumerate(df_top.head(15).iterrows()):
        log.info(
            f"  {i+1:<5} {row['name']:<20} "
            f"{row['dock_score']:>14.2f} "
            f"{row['ifp_similarity']:>6.3f} "
            f"{row['CNS_MPO']:>6.1f} "
            f"{row['composite_score']:>7.3f}"
        )
    log.info("=" * 55)
    log.info("\nSiguiente paso:")
    log.info("  python scripts/admet_and_prioritization.py \\")
    log.info("      --hits results/hits_top5pct.csv \\")
    log.info("      --output results/ --figures figures/")


if __name__ == "__main__":
    main()
