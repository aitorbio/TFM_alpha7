"""
docking_ecd_selectivity.py — Análisis de selectividad TMD vs ECD
================================================================
Dockea los candidatos que pasaron el cribado TMD contra el sitio
ortostérico del ECD para evaluar si muestran selectividad de sitio.

Lógica:
  1. Lee screening_all.csv (candidatos que pasaron el filtro TMD)
  2. Localiza los PDBQT de entrada ya generados en pdbqt_library/
  3. Dockea contra el receptor ECD con grid centrado en el sitio
     ortostérico (centroide de epibatidina, EPJ, cadena A en 8V82)
  4. Compara dock_score TMD vs ECD → selectividad de sitio
  5. Genera figura y CSV de resultados

El grid ECD se centra en el sitio ortostérico de epibatidina:
  Centro: (150.67, 159.73, 175.91)  — derivado del centroide de EPJ
  en la cadena A de PDB 8V82 (Burke et al., 2024).

Uso:
python docking_ecd_selectivity.py \
    --screening results/screening_all.csv \
    --pdbqt-dir results/pdbqt_library \
    --receptor-ecd data/8V82_ECD_prepared.pdbqt \
    --output results/ --figures figures/

Dependencias: AutoDock Vina 1.2.0, pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Grid para el sitio ortostérico del ECD
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GridConfig:
    """Grid centrado en el sitio ortostérico ECD (epibatidina, cadena A)."""

    center_x: float = 150.67  # Centroide de EPJ en 8V82 cadena A
    center_y: float = 159.73
    center_z: float = 175.91
    size_x: float = 24.0  # Å — mismo tamaño que el grid TMD
    size_y: float = 24.0
    size_z: float = 24.0
    exhaustiveness: int = 32  # Consistente con el cribado TMD
    n_poses: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Docking con Vina
# ─────────────────────────────────────────────────────────────────────────────


def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    grid: GridConfig,
) -> float | None:
    """Ejecuta Vina y devuelve el mejor score (kcal/mol) o None."""
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    # Parsear score del stdout
    for line in result.stdout.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                continue

    # Alternativa: parsear PDBQT de salida
    if Path(output_pdbqt).exists():
        with open(output_pdbqt) as f:
            for line in f:
                if "VINA RESULT" in line:
                    for p in line.split():
                        try:
                            val = float(p)
                            if val < 0:
                                return val
                        except ValueError:
                            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cribado paralelo
# ─────────────────────────────────────────────────────────────────────────────


def run_ecd_screening(
    receptor_ecd: str,
    ligands: list[tuple[str, str]],
    grid: GridConfig,
    out_dir: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Docking paralelo contra el receptor ECD."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    n = len(ligands)
    workers = max_workers or os.cpu_count() or 4

    log.info(f"Docking ECD: {n} compuestos con {workers} procesos...")
    log.info(
        f"  Grid ECD: center=({grid.center_x:.2f}, {grid.center_y:.2f}, "
        f"{grid.center_z:.2f})  size={grid.size_x}×{grid.size_y}×{grid.size_z} Å"
    )

    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for name, lig_path in ligands:
            out_path = str(Path(out_dir) / f"{name}_ecd_out.pdbqt")
            fut = executor.submit(run_vina, receptor_ecd, lig_path, out_path, grid)
            futures[fut] = (name, out_path)

        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            name, out_path = futures[fut]
            score = fut.result()
            results.append(
                {
                    "name": name,
                    "dock_score_ecd": score if score is not None else np.nan,
                    "pdbqt_out_ecd": out_path if score is not None else "",
                }
            )
            completed += 1
            if completed % 10 == 0:
                n_ok = sum(1 for r in results if not np.isnan(r["dock_score_ecd"]))
                log.info(f"  Progreso: {completed}/{n} ({n_ok} con score)")

    df = pd.DataFrame(results)
    n_ok = df["dock_score_ecd"].notna().sum()
    log.info(f"Docking ECD completado: {n_ok}/{n} con score válido")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Figura de selectividad
# ─────────────────────────────────────────────────────────────────────────────


def plot_selectivity(df: pd.DataFrame, out_path: str) -> None:
    """Figura comparativa TMD vs ECD."""
    df_valid = df.dropna(subset=["dock_score", "dock_score_ecd"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle(
        f"Análisis de selectividad TMD vs ECD\n"
        f"n = {len(df_valid)} compuestos con score en ambos sitios",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # 1. Scatter TMD vs ECD
    ax1 = axes[0]
    ax1.scatter(
        df_valid["dock_score"],
        df_valid["dock_score_ecd"],
        c="#1976D2",
        alpha=0.4,
        s=30,
        edgecolors="white",
        linewidths=0.5,
        label="Otros candidatos (n=82)"
    )
    
    # Resaltar finalistas
    if "is_final" in df_valid.columns:
        df_final = df_valid[df_valid["is_final"]]
        ax1.scatter(
            df_final["dock_score"],
            df_final["dock_score_ecd"],
            c="#E91E63",
            alpha=1.0,
            s=80,
            edgecolors="black",
            linewidths=1.2,
            label="Finalistas ADMET (n=9)",
            zorder=10
        )
    lims = [
        min(df_valid["dock_score"].min(), df_valid["dock_score_ecd"].min()) - 0.5,
        max(df_valid["dock_score"].max(), df_valid["dock_score_ecd"].max()) + 0.5,
    ]
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Diagonal (sin selectividad)")
    ax1.set_xlabel("Docking score TMD (kcal/mol)", fontsize=10)
    ax1.set_ylabel("Docking score ECD (kcal/mol)", fontsize=10)
    ax1.set_title("TMD vs ECD scores", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # Anotar zona de selectividad
    ax1.fill_between(
        lims, lims, [lims[1], lims[1]], alpha=0.05, color="green", label="Selectivo TMD"
    )
    ax1.fill_between(
        lims, [lims[0], lims[0]], lims, alpha=0.05, color="red", label="Selectivo ECD"
    )
    ax1.legend(fontsize=7, loc="upper left")

    # 2. Distribución de ΔScore
    ax2 = axes[1]
    delta = df_valid["dock_score"] - df_valid["dock_score_ecd"]
    ax2.hist(delta, bins=15, color="#1976D2", alpha=0.3, edgecolor="white", label="Frecuencia total")
    
    if "is_final" in df_valid.columns:
        delta_final = df_final["dock_score"] - df_final["dock_score_ecd"]
        ax2.hist(delta_final, bins=10, color="#E91E63", alpha=0.8, edgecolor="white", label="Finalistas ADMET")

    ax2.axvline(0, color="black", ls="--", lw=1.5, label="Sin diferencia")
    median_d = delta.median()
    ax2.axvline(
        median_d, color="#1976D2", ls="-", lw=2, label=f"Mediana Total = {median_d:.2f}"
    )
    ax2.set_xlabel("ΔScore (TMD − ECD) kcal/mol", fontsize=10)
    ax2.set_ylabel("Frecuencia", fontsize=10)
    ax2.set_title("Distribución ΔScore", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    # Interpretación
    n_tmd_better = (delta < 0).sum()
    n_ecd_better = (delta > 0).sum()
    ax2.text(
        0.98,
        0.95,
        f"TMD mejor: {n_tmd_better} ({100*n_tmd_better/len(delta):.0f}%)\n"
        f"ECD mejor: {n_ecd_better} ({100*n_ecd_better/len(delta):.0f}%)",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=9,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # 3. Resumen por cuartiles
    ax3 = axes[2]
    ax3.axis("off")
    stats = [
        ("Compuestos analizados", f"{len(df_valid)}"),
        ("Score TMD medio", f"{df_valid['dock_score'].mean():.2f} kcal/mol"),
        ("Score ECD medio", f"{df_valid['dock_score_ecd'].mean():.2f} kcal/mol"),
        ("ΔScore mediana", f"{median_d:.2f} kcal/mol"),
        ("% selectivos TMD", f"{100*n_tmd_better/len(delta):.0f}%"),
        ("", ""),
        (
            "Interpretación:",
            (
                "ΔScore < 0 → mayor afinidad TMD (selectivo)"
                if median_d < 0
                else "ΔScore > 0 → mayor afinidad ECD (no selectivo)"
            ),
        ),
    ]
    y = 0.95
    for label, value in stats:
        if not label:
            y -= 0.06
            continue
        col = (
            "#1B5E20"
            if "selectivo" in value.lower() and "no" not in value.lower()
            else "#333"
        )
        weight = "bold" if "Interpretación" in label else "normal"
        ax3.text(
            0.05,
            y,
            label,
            ha="left",
            va="top",
            fontsize=10,
            fontweight=weight,
            transform=ax3.transAxes,
            color="#555",
        )
        ax3.text(
            0.95,
            y,
            value,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            transform=ax3.transAxes,
            color=col,
        )
        y -= 0.10

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de selectividad TMD vs ECD — Docking de candidatos "
        "contra el sitio ortostérico del dominio extracelular"
    )
    ap.add_argument(
        "--screening",
        "-s",
        required=True,
        help="screening_all.csv (candidatos del cribado TMD)",
    )
    ap.add_argument(
        "--pdbqt-dir",
        required=True,
        help="Directorio con los PDBQT de entrada (results/pdbqt_library)",
    )
    ap.add_argument(
        "--receptor-ecd",
        required=True,
        help="Receptor ECD en formato PDBQT (data/8V82_ECD_prepared.pdbqt)",
    )
    ap.add_argument("--output", "-o", default="results")
    ap.add_argument("--figures", "-f", default="figures")
    ap.add_argument(
        "--workers", type=int, default=None, help="Cores para Vina (default: todos)"
    )
    ap.add_argument("--exhaustiveness", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Leer candidatos TMD ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("ANÁLISIS DE SELECTIVIDAD TMD vs ECD")
    log.info("=" * 60)

    df = pd.read_csv(args.screening)
    log.info(f"\n[1/4] Candidatos cargados: {len(df)} compuestos")

    # ── 2. Localizar PDBQT de entrada ────────────────────────────────────────
    log.info(f"\n[2/4] Buscando PDBQT en {args.pdbqt_dir}...")
    pdbqt_dir = Path(args.pdbqt_dir)
    ligands = []
    not_found = []

    for _, row in df.iterrows():
        name = str(row["name"])
        pdbqt_path = pdbqt_dir / f"{name}.pdbqt"
        if pdbqt_path.exists():
            ligands.append((name, str(pdbqt_path)))
        else:
            not_found.append(name)

    log.info(f"  Encontrados: {len(ligands)} PDBQT")
    if not_found:
        log.warning(f"  No encontrados: {len(not_found)} (se omitirán)")
        if len(not_found) <= 5:
            for nf in not_found:
                log.warning(f"    - {nf}")

    if not ligands:
        sys.exit("Error: No se encontraron PDBQT de entrada para ningún candidato")

    # ── 3. Docking ECD ───────────────────────────────────────────────────────
    log.info(f"\n[3/4] Docking contra ECD...")
    grid = GridConfig(exhaustiveness=args.exhaustiveness)

    df_ecd = run_ecd_screening(
        args.receptor_ecd,
        ligands,
        grid,
        str(out_dir / "docking_poses_ecd"),
        max_workers=args.workers,
    )

    # ── 4. Merge y análisis ──────────────────────────────────────────────────
    log.info(f"\n[4/4] Analizando selectividad...")
    df["name"] = df["name"].astype(str)
    df_ecd["name"] = df_ecd["name"].astype(str)
    df_merged = df.merge(df_ecd, on="name", how="left")

    # Cargar finalistas para resaltar
    final_file = "results/candidates_strict_admet_passed.csv"
    if os.path.exists(final_file):
        df_final_list = pd.read_csv(final_file)
        final_names = set(df_final_list["name"].astype(str))
        df_merged["is_final"] = df_merged["name"].apply(lambda x: x in final_names)
    else:
        df_merged["is_final"] = False

    # Estadísticas
    df_valid = df_merged.dropna(subset=["dock_score", "dock_score_ecd"])
    n_valid = len(df_valid)

    if n_valid == 0:
        log.warning("No hay compuestos con score válido en ambos sitios.")
        df_merged.to_csv(out_dir / "selectivity_ecd.csv", index=False)
        return

    delta = df_valid["dock_score"] - df_valid["dock_score_ecd"]
    n_tmd_better = (delta < 0).sum()

    log.info(f"\n{'='*60}")
    log.info(f"RESULTADOS DE SELECTIVIDAD")
    log.info(f"{'='*60}")
    log.info(f"  Compuestos analizados:   {n_valid}")
    log.info(f"  Score TMD medio:         {df_valid['dock_score'].mean():.2f} kcal/mol")
    log.info(
        f"  Score ECD medio:         {df_valid['dock_score_ecd'].mean():.2f} kcal/mol"
    )
    log.info(f"  ΔScore mediana:          {delta.median():.2f} kcal/mol")
    log.info(
        f"  Selectivos TMD (mejor en TMD): {n_tmd_better}/{n_valid} "
        f"({100*n_tmd_better/n_valid:.0f}%)"
    )
    log.info(f"{'='*60}")

    if delta.median() < 0:
        log.info("  ✓ Los candidatos muestran preferencia por el sitio TMD")
        log.info("    → Consistente con selectividad de sitio del protocolo")
    else:
        log.warning("  ✗ Los candidatos NO muestran preferencia clara por el TMD")
        log.warning(
            "    → Revisar si el grid ECD es correcto o si los compuestos "
            "son promiscuos"
        )

    # Guardar
    df_merged.to_csv(out_dir / "selectivity_ecd.csv", index=False)
    log.info(f"\n  Resultados guardados: {out_dir / 'selectivity_ecd.csv'}")

    # Top 10 más selectivos TMD
    df_valid_sorted = df_valid.copy()
    df_valid_sorted["delta_score"] = delta
    df_valid_sorted = df_valid_sorted.sort_values("delta_score")

    log.info(f"\nTop 10 más selectivos para TMD:")
    log.info(f"  {'Rank':<5} {'ID':<20} {'TMD':>10} {'ECD':>10} {'Δ':>8}")
    log.info(f"  {'-'*53}")
    for i, (_, row) in enumerate(df_valid_sorted.head(10).iterrows()):
        log.info(
            f"  {i+1:<5} {row['name']:<20} {row['dock_score']:>10.2f} "
            f"{row['dock_score_ecd']:>10.2f} {row['delta_score']:>8.2f}"
        )

    # Figura
    plot_selectivity(df_merged, str(fig_dir / "fig_selectivity_tmd_vs_ecd.png"))

    log.info(f"\n✓ Análisis de selectividad completado.")


if __name__ == "__main__":
    main()
