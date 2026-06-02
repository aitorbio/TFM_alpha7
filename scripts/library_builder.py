"""
library_builder.py  — Módulo 2 del pipeline para cribado de α7 nAChR
====
En esta sección se va a construir el cribado sobre la base de datos
ZINC-22, el cual permite acceder a millones de compuestos y
filtrarlos según diferentes criterios. En este caso, se va a aplicar un
filtrado por propiedades moleculares (MW, logP, TPSA, hBD, hBA)
según lo establecido en el apartado 3.4.2 del documento metodológico.


Fuente: ZINC-22 (https://zinc22.docking.org).
En ZINC-22 debemos descargar las moléculas para el cribdao ->ZINC_raw_download.csv
Filtrado: RDKit + criterios de la metodología.

Dependencias:
    rdkit >= 2023.09
    pandas, numpy, requests, matplotlib
    mol_properties_rdkit.py  (IMPORTANTE = mismo directorio)

Uso:
# Filtrar un fichero .smi
python library_builder.py \
    --input data/ZINC_raw_download.csv \
    --output results/library_filtered.csv

# Modo test (usa PAMs de referencia ya integrados en el codigo)
# Así puedo comprobar que todo funciona correctamente con pocas moléculas.
python library_builder.py --test

# He creado una funcionalidad para descargar compuestos de ZINC-22
# con el parámetro --download-zinc22. Ahora mismo la API no funciona
# correctamente, pero esta es la forma de descargar compuestos de ZINC-22.
# No es útil pero he trabajado bastante en ella hasta darme cuenta que no funcionaba
# así que la dejo como ejemplo de como funcionaría.
# Por defecto descarga 5000 compuestos.
python library_builder.py --download-zinc22
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Importamos mol_properties_rdkit. IMPORTANTE que esté en la misma carpeta.
sys.path.insert(0, str(Path(__file__).parent))
from mol_properties_rdkit import (
    batch_process,
    CNS_MPO_CRITERIA,
    FilterCriteria,
    write_report,
    read_input,
)

try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdMolDescriptors
    from rdkit.Chem import FilterCatalog
except ImportError:
    # Debería estar inslatado ya que se ha comprobado en verify_installation.py
    sys.exit("RDKit no encontrado. Instala con: conda install -c conda-forge rdkit")

try:
    import requests

    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Descarga desde ZINC-22

ZINC22_BASE = "https://zinc22.docking.org"

# Subsets de ZINC-22 adecuados para el cribado del sitio TMD α7 nAChR:
# - in-stock: disponibles inmediatamente
# - mw: 250–450 Da (filtrado previo)
# - logp: 1.5–4.5
ZINC22_QUERY_PARAMS = {
    "count": 200,  # por página (máximo por request)
    "mw_min": 250,
    "mw_max": 450,
    "logp_min": 1.5,
    "logp_max": 4.5,
    "tpsa_max": 90,
    "hbd_max": 3,
    "availability": "for-sale",
    "output_fields": "zinc_id,smiles",
}


def download_zinc22(
    n_compounds: int = 5000,
    output_path: str = "data/zinc22_raw.smi",
    delay: float = 1.0,
) -> list[tuple[str, str]]:
    """
    Descarga compuestos de ZINC-22 mediante la API REST.
    Aplica pre-filtros de MW, logP y TPSA en la query para reducir
    el volumen descargado antes de aplicar los filtros RDKit completos.

    Parámetros:
        n_compounds : número de compuestos a descargar
        output_path : fichero .smi donde se guardan los resultados crudos
        delay       : segundos entre requests (respetar rate limit ZINC-22)

    Devuelve:
        lista de (smiles, zinc_id)
    """
    if not REQUESTS_OK:
        raise ImportError("requests no instalado. pip install requests")

    log.info(f"Descargando {n_compounds} compuestos de ZINC-22...")
    compounds = []
    page = 1
    per_page = min(200, n_compounds)

    while len(compounds) < n_compounds:
        params = {**ZINC22_QUERY_PARAMS, "count": per_page, "page": page}
        try:
            resp = requests.get(
                f"{ZINC22_BASE}/substances.json",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            log.error(f"Error HTTP en página {page}: {e}")
            break
        except Exception as e:
            log.error(f"Error de conexión: {e}")
            break

        substances = data.get("substances", [])
        if not substances:
            log.info("No hay más compuestos disponibles en ZINC-22.")
            break

        for s in substances:
            zinc_id = s.get("zinc_id", f"ZINC_unknown_{len(compounds)}")
            smiles = s.get("smiles", "")
            if smiles and len(compounds) < n_compounds:
                compounds.append((smiles, zinc_id))

        log.info(
            f"  Página {page}: +{len(substances)} compuestos "
            f"(total acum. {len(compounds)})"
        )
        page += 1
        time.sleep(delay)

    # Guardar raw
    if compounds:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# ZINC-22 download — nAChR alpha7PAM screening\n")
            for smi, zid in compounds:
                f.write(f"{smi}\t{zid}\n")
        log.info(f"Raw guardado: {output_path} ({len(compounds)} compuestos)")

    return compounds


# Desduplicación por huella digital Morgan (RDKit)


def deduplicate_by_morgan(
    df: pd.DataFrame, radius: int = 2, n_bits: int = 2048, tanimoto_cutoff: float = 0.85
) -> pd.DataFrame:
    """
    Elimina duplicados estructurales mediante huella digital Morgan de forma rápida O(N).
    Cuando dos compuestos tienen Tanimoto > cutoff (0.85), conserva el de mayor CNS_MPO.
    """
    log.info(
        f"  Deduplicando {len(df)} compuestos O(N) (Tanimoto cutoff={tanimoto_cutoff})..."
    )

    if "CNS_MPO" in df.columns:
        df = df.sort_values("CNS_MPO", ascending=False).reset_index(drop=True)

    kept_indices = []
    kept_fps = []

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles_canon"])
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, n_bits)

        if not kept_fps:
            kept_fps.append(fp)
            kept_indices.append(i)
            continue

        # Comparación masiva O(N) usando C++ core de RDKit
        sims = DataStructs.BulkTanimotoSimilarity(fp, kept_fps)
        if max(sims) <= tanimoto_cutoff:
            kept_fps.append(fp)
            kept_indices.append(i)

    df_out = df.iloc[kept_indices].copy().reset_index(drop=True)
    n_removed = len(df) - len(df_out)
    log.info(
        f"  Deduplicación: {n_removed} duplicados eliminados → {len(df_out)} únicos"
    )
    return df_out


# Pipeline de filtrado completo

def build_filtered_library(
    compounds: list[tuple[str, str]],
    criteria: FilterCriteria = CNS_MPO_CRITERIA,
    deduplicate: bool = True,
    tanimoto_cutoff: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el pipeline completo de filtrado:
        1. Cálculo de propiedades con RDKit
        2. Filtros fisicoquímicos (§3.4.2)
        3. Filtros PAINS + BRENK (§3.4.3)
        4. Deduplicación por Morgan (§3.4.4)

    Devuelve (df_all, df_filtered).
    """
    log.info(f"Procesando {len(compounds)} compuestos con RDKit...")
    records = batch_process(compounds, criteria)

    df_all = pd.DataFrame([asdict(r) for r in records])

    # Filtrar válidos que superan todos los criterios
    df_pass = (
        df_all[df_all["pass_all"] & (df_all["parse_error"] == "")]
        .copy()
        .reset_index(drop=True)
    )

    log.info(f"Tras filtros fisicoquímicos + PAINS: {len(df_pass)}/{len(df_all)}")

    # Deduplicación
    if deduplicate and len(df_pass) > 1:
        df_pass = deduplicate_by_morgan(df_pass, tanimoto_cutoff=tanimoto_cutoff)

    df_pass = df_pass.sort_values("CNS_MPO", ascending=False).reset_index(drop=True)
    return df_all, df_pass


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────


def plot_filter_distributions(
    df_all: pd.DataFrame, df_pass: pd.DataFrame, out_path: str
) -> None:
    """Distribuciones antes/después de filtrado — 8 propiedades."""
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle(
        f"Módulo 2 — Filtrado fisicoquímico\n"
        f"Entrada: {len(df_all)}  →  Salida: {len(df_pass)} "
        f"({len(df_pass)/max(len(df_all),1)*100:.1f}% retenidos)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    props = [
        ("MW", "Peso Molecular (Da)", (200, 520), (250, 450), "#2196F3"),
        ("clogP", "clogP (RDKit)", (-1, 7), (1.5, 4.5), "#4CAF50"),
        ("TPSA", "TPSA (Å²)", (0, 160), (0, 90), "#FF9800"),
        ("HBD", "Donadores H-bond", (-0.5, 6), (-0.5, 3.5), "#9C27B0"),
        ("HBA", "Aceptores H-bond", (-0.5, 14), (-0.5, 7.5), "#F44336"),
        ("RotBonds", "Nº enlaces rotativos", (-0.5, 12), (-0.5, 8.5), "#00BCD4"),
        ("ArRings", "Nº anillos aromát.", (-0.5, 6), (0.5, 4.5), "#795548"),
        ("CNS_MPO", "CNS MPO score", (0, 6.5), (4, 6.5), "#E91E63"),
    ]
    grey = "#ADB5BD"

    for idx, (prop, label, xlim, ok, color) in enumerate(props):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        bins = np.linspace(xlim[0], xlim[1], 25)
        ax.hist(
            df_all[prop],
            bins=bins,
            color=grey,
            alpha=0.55,
            label="Todas",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.hist(
            df_pass[prop],
            bins=bins,
            color=color,
            alpha=0.85,
            label="Filtradas",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvspan(ok[0], ok[1], alpha=0.07, color=color)
        ax.axvline(ok[0], color=color, ls="--", lw=1.2, alpha=0.7)
        if ok[1] < xlim[1]:
            ax.axvline(ok[1], color=color, ls="--", lw=1.2, alpha=0.7)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        ax.set_xlim(xlim)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


def plot_filter_cascade(df_all: pd.DataFrame, out_path: str) -> None:
    """Cascada de filtrado — compuestos eliminados por cada criterio."""
    filter_labels = [
        "Entrada",
        "MW\n250–450",
        "clogP\n1.5–4.5",
        "TPSA\n<90 Å²",
        "HBD ≤ 3",
        "HBA ≤ 7",
        "RotBonds ≤ 8",
        "ArRings\n1–4",
        "PAINS",
        "BRENK",
    ]
    filter_cols = [
        None,
        "pass_MW",
        "pass_clogP",
        "pass_TPSA",
        "pass_HBD",
        "pass_HBA",
        "pass_RotBonds",
        "pass_ArRings",
        "pass_PAINS",
        "pass_BRENK",
    ]
    counts = [len(df_all)]
    cur = df_all[df_all["parse_error"] == ""].copy()
    for col in filter_cols[1:]:
        cur = cur[cur[col]]
        counts.append(len(cur))

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    colors = ["#1565C0"] + ["#42A5F5"] * (len(counts) - 2) + ["#2E7D32"]
    bars = ax.bar(
        range(len(counts)),
        counts,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.7,
    )

    for i, (bar, c) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(c),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        if i > 0 and counts[i - 1] - c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"−{counts[i-1]-c}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(range(len(filter_labels)))
    ax.set_xticklabels(filter_labels, fontsize=8.5)
    ax.set_ylabel("Compuestos retenidos", fontsize=10)
    ax.set_title(
        "Cascada de filtrado fisicoquímico — §3.4", fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, max(counts) * 1.15)
    ax.plot(range(len(counts)), counts, "k--o", ms=5, lw=1, alpha=0.35)
    pct = counts[-1] / counts[0] * 100 if counts[0] else 0
    ax.text(
        len(counts) - 1,
        counts[-1] + max(counts) * 0.04,
        f"{pct:.0f}%\nretenidos",
        ha="center",
        fontsize=9,
        color="#2E7D32",
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    ap = argparse.ArgumentParser(description="Módulo 2 — Biblioteca de cribado ZINC-22")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="Fichero .smi / .csv ya descargado")
    src.add_argument(
        "--download", action="store_true", help="Descargar desde ZINC-22 (requiere red)"
    )
    src.add_argument(
        "--test", action="store_true", help="Modo test con PAMs de referencia"
    )
    ap.add_argument(
        "--n-download",
        type=int,
        default=5000,
        help="Nº compuestos a descargar de ZINC-22 (default: 5000)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="results",
        help="Directorio de salida (default: results/)",
    )
    ap.add_argument(
        "--figures", default="figures", help="Directorio de figuras (default: figures/)"
    )
    ap.add_argument(
        "--no-dedup", action="store_true", help="Desactivar deduplicación Morgan"
    )
    ap.add_argument(
        "--tanimoto",
        type=float,
        default=0.85,
        help="Umbral Tanimoto para deduplicación (default: 0.85)",
    )
    ap.add_argument("--mw-min", type=float, default=250.0)
    ap.add_argument("--mw-max", type=float, default=450.0)
    ap.add_argument("--no-brenk", action="store_true")
    return ap.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    criteria = FilterCriteria(
        mw_min=args.mw_min, mw_max=args.mw_max, remove_brenk=not args.no_brenk
    )

    # ── Cargar compuestos ─────────────────────────────────────────────────
    if args.test:
        from mol_properties_rdkit import TEST_COMPOUNDS

        compounds = [(smi, name) for name, smi in TEST_COMPOUNDS]
        log.info(f"Modo test: {len(compounds)} compuestos")

    elif args.download:
        raw_path = str(out_dir / "zinc22_raw.smi")
        compounds = download_zinc22(args.n_download, raw_path)
        if not compounds:
            log.error("Descarga fallida. Comprueba conexión a ZINC-22.")
            sys.exit(1)

    else:  # --input
        compounds = read_input(args.input)
        log.info(f"{len(compounds)} compuestos leídos de {args.input}")

    # ── Filtrado ─────────────────────────────────────────────────────────
    df_all, df_pass = build_filtered_library(
        compounds,
        criteria,
        deduplicate=not args.no_dedup,
        tanimoto_cutoff=args.tanimoto,
    )

    # ── Guardar ──────────────────────────────────────────────────────────
    path_all = out_dir / "library_all.csv"
    path_pass = out_dir / "library_filtered.csv"
    df_all.to_csv(path_all, index=False)
    df_pass.to_csv(path_pass, index=False)

    # Generar informe desde DataFrames
    n_err = (df_all["parse_error"] != "").sum()
    n_valid = len(df_all) - n_err
    lines = [
        "=" * 60,
        f"INFORME — Módulo 2 Biblioteca de cribado α7 nAChR",
        "=" * 60,
        f"  Entrada:               {len(df_all):>7}",
        f"  Errores parseo:        {n_err:>7}",
        f"  Válidas:               {n_valid:>7}",
        f"  Tras filtros:          {len(df_pass):>7}  "
        f"({len(df_pass)/max(n_valid,1)*100:.1f}%)",
        f"  CNS MPO ≥ 4:          "
        f"{(df_pass['CNS_MPO']>=4).sum():>7}  "
        f"({(df_pass['CNS_MPO']>=4).sum()/max(len(df_pass),1)*100:.1f}%)",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    (out_dir / "filter_report.txt").write_text(report_text)
    print(report_text)

    # ── Figuras ───────────────────────────────────────────────────────────
    if len(df_pass) > 0:
        plot_filter_distributions(
            df_all[df_all["parse_error"] == ""],
            df_pass,
            str(fig_dir / "fig_filter_distributions.png"),
        )
        plot_filter_cascade(
            df_all[df_all["parse_error"] == ""],
            str(fig_dir / "fig_filter_cascade.png"),
        )

    log.info(f"✓ Módulo 2 completado. Biblioteca lista: {path_pass}")
    log.info(f"  {len(df_pass)} compuestos para docking.")


if __name__ == "__main__":
    main()
