"""
admet_import_results.py
=======================
Paso 2 del flujo manual: importa los CSV descargados de SwissADME,
pkCSM y ADMETlab 2.0, consolida los resultados, aplica los criterios
de aceptación y genera la priorización final.

Uso:
python admet_import_results.py \
    --input     admet_inputs/ \
    --hits      results/hits_top5pct.csv \
    --output    results/ \
    --figures   figures/

Ficheros esperados en --input:
    02_swissadme_results.csv   (exportado desde SwissADME)
    02_pkcsm_results.csv       (exportado desde pkCSM Batch)
    02_admetlab_results.csv    (exportado desde ADMETlab 2.0)
    (cualquiera de los tres es suficiente para continuar)

Columnas mínimas necesarias:
    SwissADME : Molecule, SMILES, BBB (BOILED-Egg), Pgp substrate,
                iLOGP, WLOGP, TPSA, ...
    pkCSM     : Molecule, Blood-Brain-Barrier, P-glycoprotein substrate,
                hERG I inhibitor, hERG II inhibitor, Half life,
                Hepatotoxicity
    ADMETlab  : SMILES o Name, DILI, AMES, hERG, BBB, P-gp substrate
"""

from __future__ import annotations

import argparse
import logging
import math
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
# Parsers para cada plataforma
# ─────────────────────────────────────────────────────────────────────────────


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Devuelve el primer nombre de columna que existe en df (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _to_bool(val) -> bool | None:
    """Convierte un valor de plataforma web a bool o None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in (
        "yes",
        "true",
        "1",
        "substrate",
        "inhibitor",
        "positive",
        "mutagen",
        "high risk",
        "high",
    ):
        return True
    if s in (
        "no",
        "false",
        "0",
        "non-substrate",
        "non-inhibitor",
        "negative",
        "non-mutagen",
        "low risk",
        "low",
        "medium",
    ):
        return False
    return None


def parse_swissadme(path: str) -> pd.DataFrame:
    """
    Parsea el CSV de SwissADME.
    Columnas clave exportadas por SwissADME:
      Molecule, SMILES, MW, TPSA, iLOGP, XLOGP3, WLOGP, MLOGP,
      Silicos-IT logP, Consensus Log Po/w,
      GI absorption, BBB permeant, Pgp substrate,
      CYP1A2 inhibitor, CYP2C19 inhibitor, CYP2C9 inhibitor,
      CYP2D6 inhibitor, CYP3A4 inhibitor,
      Log Kp (skin permeation), Lipinski, Ghose, ...
    """
    df = pd.read_csv(path)
    log.info(f"  SwissADME: {len(df)} filas, columnas: {list(df.columns[:8])}...")

    name_col = _col(df, "Molecule", "Name", "name", "ID")
    bbb_col = _col(df, "BBB permeant", "BBB", "bbb permeant")
    pgp_col = _col(df, "Pgp substrate", "P-gp substrate", "pgp")
    logp_col = _col(df, "Consensus Log Po/w", "iLOGP", "WLOGP", "clogp")

    result = pd.DataFrame()
    if name_col:
        result["name"] = df[name_col].astype(str)
    else:
        result["name"] = [f"mol_{i}" for i in range(len(df))]

    # BBB: "Yes"/"No" en SwissADME
    if bbb_col:
        result["BBB_swissadme"] = df[bbb_col].apply(_to_bool)
        result["BBB_ok_sw"] = result["BBB_swissadme"]
    else:
        log.warning("  SwissADME: columna BBB no encontrada")
        result["BBB_ok_sw"] = None

    # P-gp: "Yes"/"No" — queremos No sustrato → Pgp_ok = NOT substrate
    if pgp_col:
        result["Pgp_substrate_sw"] = df[pgp_col].apply(_to_bool)
        result["Pgp_ok_sw"] = result["Pgp_substrate_sw"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        log.warning("  SwissADME: columna P-gp no encontrada")
        result["Pgp_ok_sw"] = None

    if logp_col:
        result["clogP_sw"] = pd.to_numeric(df[logp_col], errors="coerce")

    return result


def parse_pkcsm(path: str) -> pd.DataFrame:
    """
    Parsea el CSV de pkCSM Batch Prediction.
    Columnas clave:
      Molecule, Blood-Brain-Barrier (numeric logBB),
      P-glycoprotein substrate (Yes/No),
      hERG I inhibitor (Yes/No), hERG II inhibitor (Yes/No),
      Half life (h → convertimos a min),
      Hepatotoxicity (Yes/No)
    """
    # pkCSM exporta TSV (tabuladores) aunque la extensión sea .csv
    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) == 1:  # si solo hay 1 columna, intentar con coma
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    log.info(f"  pkCSM: {len(df)} filas × {len(df.columns)} columnas")
    log.info(f"  Columnas pkCSM (primeras 6): {list(df.columns[:6])}...")

    # pkCSM real column names (verified from actual export):
    #   SMILES, BBB permeability, P-glycoprotein substrate,
    #   hERG II inhibitor, AMES toxicity, Hepatotoxicity
    name_col = _col(df, "SMILES", "Molecule", "Name", "name", "id")
    bbb_col = _col(df, "BBB permeability", "Blood-Brain-Barrier", "BBB", "logBB", "bbb")
    pgp_col = _col(
        df,
        "P-glycoprotein substrate",
        "Pgp substrate",
        "P-gp substrate",
        "pgp_substrate",
    )
    herg_col = _col(
        df, "hERG II inhibitor", "hERG I inhibitor", "hERG inhibitor", "herg"
    )
    half_col = _col(df, "Half life", "Half-life", "t_half", "T1/2")
    hepato = _col(df, "Hepatotoxicity", "hepatotox", "DILI")
    ames_col_pk = _col(df, "AMES toxicity", "AMES", "ames", "Ames")
    log.info(
        f"  Columnas detectadas: "
        f"BBB={'✓ ('+bbb_col+')' if bbb_col else '✗'} | "
        f"PGP={'✓' if pgp_col else '✗'} | "
        f"hERG={'✓' if herg_col else '✗'} | "
        f"AMES={'✓ ('+ames_col_pk+')' if ames_col_pk else '✗'} | "
        f"Hepato={'✓' if hepato else '✗'}"
    )

    result = pd.DataFrame()
    result["name"] = (
        df[name_col].astype(str) if name_col else [f"mol_{i}" for i in range(len(df))]
    )

    # logBB (numérico — positivo = buena permeabilidad)
    if bbb_col:
        result["BBB_logBB"] = pd.to_numeric(df[bbb_col], errors="coerce")
        result["BBB_ok_pk"] = result["BBB_logBB"] > 0
    else:
        log.warning(
            "  pkCSM: columna BBB no encontrada — verifica que el CSV tiene columna 'BBB permeability'"
        )
        result["BBB_ok_pk"] = None

    # P-gp
    if pgp_col:
        result["Pgp_substrate_pk"] = df[pgp_col].apply(_to_bool)
        result["Pgp_ok_pk"] = result["Pgp_substrate_pk"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        result["Pgp_ok_pk"] = None

    # hERG: queremos que NO sea inhibidor
    if herg_col:
        result["hERG_inhibitor_pk"] = df[herg_col].apply(_to_bool)
        result["hERG_ok_pk"] = result["hERG_inhibitor_pk"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        result["hERG_ok_pk"] = None

    # t½ microsomal: pkCSM puede tener "Half life" o no tenerlo
    # Si no existe, usar "Total Clearance" como proxy inverso:
    #   clearance < 2 L/h/kg ≈ t½ razonable para candidatos SNC
    clearance_col = _col(df, "Total Clearance", "Clearance", "clearance")
    if half_col:
        result["t_half_min_pk"] = pd.to_numeric(df[half_col], errors="coerce") * 60
        result["Microsomal_ok_pk"] = result["t_half_min_pk"] > 30
    elif clearance_col:
        cl = pd.to_numeric(df[clearance_col], errors="coerce")
        result["t_half_min_pk"] = (0.693 / (cl + 1e-6)) * 60  # proxy
        result["Microsomal_ok_pk"] = cl < 10  # clearance < 10 = aceptable
        log.info("  Microsomal: usando 'Total Clearance' como proxy de t½")
    else:
        result["Microsomal_ok_pk"] = None

    # Hepatotoxicidad
    if hepato:
        result["Hepatotox_pk"] = df[hepato].apply(_to_bool)
        result["DILI_ok_pk"] = result["Hepatotox_pk"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        result["DILI_ok_pk"] = None

    # AMES: queremos negativo (no mutagénico) — "No" = seguro
    if ames_col_pk:
        result["AMES_pk"] = df[ames_col_pk].apply(_to_bool)
        result["AMES_ok_pk"] = result["AMES_pk"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        result["AMES_ok_pk"] = None

    return result


def parse_admetlab(path: str) -> pd.DataFrame:
    """
    Parsea el CSV de ADMETlab 2.0.
    Columnas clave (los nombres pueden variar según versión):
      Name/SMILES, DILI, AMES, hERG inhibition,
      BBB permeability, P-gp substrate, BCRP substrate,
      Respiratory toxicity, Skin sensitization
    """
    df = pd.read_csv(path)
    log.info(f"  ADMETlab: {len(df)} filas, columnas: {list(df.columns[:8])}...")

    name_col = _col(df, "Name", "Molecule", "name", "ID", "SMILES")
    dili_col = _col(df, "DILI", "Drug Induced Liver Injury", "dili", "DILI_label")
    ames_col = _col(
        df, "AMES", "AMES mutagenicity", "ames", "AMES_label", "Ames mutagenicity"
    )
    herg_col = _col(df, "hERG", "hERG inhibition", "hERG_label", "hERG_pIC50", "herg")
    bbb_col = _col(
        df, "BBB", "BBB permeability", "bbb", "BBB_label", "Blood Brain Barrier"
    )
    pgp_col = _col(df, "P-gp substrate", "Pgp", "pgp", "P-glycoprotein substrate")
    bcrp_col = _col(df, "BCRP substrate", "BCRP", "bcrp")

    result = pd.DataFrame()
    result["name"] = (
        df[name_col].astype(str) if name_col else [f"mol_{i}" for i in range(len(df))]
    )

    # DILI: en ADMETlab3 es probabilidad [0–1]
    # p < 0.5 = bajo riesgo hepatotóxico
    if dili_col:
        numeric_dili = pd.to_numeric(df[dili_col], errors="coerce")
        if numeric_dili.notna().sum() > len(df) * 0.3:  # es numérico
            result["DILI_prob"] = numeric_dili
            result["DILI_risk_al"] = numeric_dili.apply(
                lambda x: (
                    "Low"
                    if (not pd.isna(x) and x < 0.5)
                    else ("High" if not pd.isna(x) else "Unknown")
                )
            )
            result["DILI_ok_al"] = numeric_dili < 0.5
        else:  # es texto

            def parse_dili(v):
                s = str(v).strip().lower()
                if any(x in s for x in ("low", "no", "negative")):
                    return "Low"
                if any(x in s for x in ("high", "yes", "positive")):
                    return "High"
                return str(v)

            result["DILI_risk_al"] = df[dili_col].apply(parse_dili)
            result["DILI_ok_al"] = result["DILI_risk_al"] == "Low"
    else:
        log.warning("  ADMETlab: columna DILI no encontrada")
        result["DILI_ok_al"] = None

    # AMES: en ADMETlab3 es probabilidad [0–1] (columna "Ames")
    # p < 0.5 = no mutagénico (seguro); p > 0.5 = mutagénico (rechazar)
    if ames_col:
        numeric_ames = pd.to_numeric(df[ames_col], errors="coerce")
        if numeric_ames.notna().sum() > len(df) * 0.3:  # es numérico
            result["AMES_prob"] = numeric_ames
            result["AMES_ok_al"] = numeric_ames < 0.5
        else:  # es texto Yes/No
            result["AMES_al"] = df[ames_col].apply(_to_bool)
            result["AMES_ok_al"] = result["AMES_al"].map(
                lambda x: (not x) if x is not None else None
            )
    else:
        result["AMES_ok_al"] = None

    # hERG: puede ser pIC50 numérico o Yes/No
    if herg_col:
        vals = df[herg_col]
        numeric = pd.to_numeric(vals, errors="coerce")
        if numeric.notna().sum() > len(vals) * 0.5:
            result["hERG_pIC50_al"] = numeric
            result["hERG_ok_al"] = numeric < 6.0
        else:
            result["hERG_inhibitor_al"] = vals.apply(_to_bool)
            result["hERG_ok_al"] = result["hERG_inhibitor_al"].map(
                lambda x: (not x) if x is not None else None
            )
    else:
        result["hERG_ok_al"] = None

    # BBB
    if bbb_col:
        vals = df[bbb_col]
        numeric = pd.to_numeric(vals, errors="coerce")
        if numeric.notna().sum() > len(vals) * 0.5:
            result["BBB_logBB_al"] = numeric
            result["BBB_ok_al"] = numeric > 0
        else:
            result["BBB_al"] = vals.apply(_to_bool)
            result["BBB_ok_al"] = result["BBB_al"]
    else:
        result["BBB_ok_al"] = None

    # P-gp
    if pgp_col:
        result["Pgp_al"] = df[pgp_col].apply(_to_bool)
        result["Pgp_ok_al"] = result["Pgp_al"].map(
            lambda x: (not x) if x is not None else None
        )
    else:
        result["Pgp_ok_al"] = None

    # BCRP: en ADMETlab3 es probabilidad [0–1] (columna "BCRP")
    # p < 0.5 = no sustrato (seguro); valores muy bajos (<0.05) son muy seguros
    if bcrp_col:
        numeric_bcrp = pd.to_numeric(df[bcrp_col], errors="coerce")
        if numeric_bcrp.notna().sum() > len(df) * 0.3:
            result["BCRP_prob"] = numeric_bcrp
            result["BCRP_ok_al"] = numeric_bcrp < 0.5
        else:
            result["BCRP_al"] = df[bcrp_col].apply(_to_bool)
            result["BCRP_ok_al"] = result["BCRP_al"].map(
                lambda x: (not x) if x is not None else None
            )
    else:
        result["BCRP_ok_al"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Consolidación por mayoría
# ─────────────────────────────────────────────────────────────────────────────


def majority_vote(*values) -> bool | None:
    """
    Voto por mayoría simple entre valores booleanos.
    Ignora None. Si hay empate o todos son None, devuelve None.
    """
    known = [v for v in values if v is not None]
    if not known:
        return None
    n_true = sum(known)
    if n_true > len(known) / 2:
        return True
    if n_true < len(known) / 2:
        return False
    return None  # empate


def consolidate(
    df_sw: pd.DataFrame | None,
    df_pk: pd.DataFrame | None,
    df_al: pd.DataFrame | None,
    df_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida los resultados de las tres plataformas mediante voto por mayoría.

    Estrategia de unión (en orden de preferencia):
      1. Por posición (índice de fila): se usa cuando los CSV de las webs
         tienen el mismo número de filas que df_ref y están en el mismo orden.
         Es el caso habitual: se exportan los SMILES en orden y se suben a las
         webs en ese mismo orden.
      2. Por SMILES canónico (RDKit): si los conteos difieren (ej. pkCSM falló
         con una molécula), se normaliza el SMILES con RDKit y se une por él.
      3. Por nombre ZINC: fallback final si los anteriores no funcionan.
    """
    base = df_ref.copy()
    base["name"] = base["name"].astype(str)
    base = base.reset_index(drop=True)

    def _join(
        base_df: pd.DataFrame, admet_df: pd.DataFrame, suffix: str
    ) -> pd.DataFrame:
        """Une df_ref con un DataFrame ADMET por la mejor estrategia disponible."""
        admet_df = admet_df.copy().reset_index(drop=True)

        n_ref = len(base_df)
        n_admet = len(admet_df)

        # ── Estrategia 1: por posición ─────────────────────────────────────
        if n_admet == n_ref:
            log.info(f"    Unión por POSICIÓN ({n_admet} filas coinciden)")
            admet_df.index = base_df.index
            joined = base_df.join(admet_df, rsuffix=suffix)
            return joined

        # ── Estrategia 2: por SMILES canónico ─────────────────────────────
        log.info(
            f"    Unión por SMILES ({n_ref} ref vs {n_admet} admet — "
            f"faltan {n_ref - n_admet} moléculas en esta plataforma)"
        )

        # Columna SMILES en df_ref
        smi_ref_col = next(
            (
                c
                for c in ["smiles_canon", "smiles_input", "smiles", "SMILES"]
                if c in base_df.columns
            ),
            None,
        )
        # Columna SMILES en admet_df
        smi_admet_col = next(
            (
                c
                for c in [
                    "smiles_canon",
                    "smiles",
                    "Canonical SMILES",
                    "SMILES",
                    "raw_smiles",
                ]
                if c in admet_df.columns
            ),
            None,
        )

        if smi_ref_col and smi_admet_col:
            try:
                from rdkit import Chem

                def canon(s):
                    m = Chem.MolFromSmiles(str(s))
                    return Chem.MolToSmiles(m) if m else str(s).strip()

                base_df = base_df.copy()
                admet_df = admet_df.copy()
                base_df["_smi_key"] = base_df[smi_ref_col].apply(canon)
                admet_df["_smi_key"] = admet_df[smi_admet_col].apply(canon)
                joined = base_df.merge(
                    admet_df, on="_smi_key", how="left", suffixes=("", suffix)
                )
                joined = joined.drop(columns=["_smi_key"], errors="ignore")
                matched = (
                    joined[admet_df.columns[0] + (suffix if suffix else "")]
                    .notna()
                    .sum()
                    if (admet_df.columns[0] + suffix) in joined.columns
                    else "?"
                )
                log.info(
                    f"    Coincidencias SMILES: ~{n_admet - (n_ref - n_admet)} / {n_ref}"
                )
                return joined
            except Exception as e:
                log.warning(
                    f"    Unión por SMILES falló ({e}), usando posición parcial"
                )

        # ── Estrategia 3: por posición parcial (las primeras n_admet filas) ─
        log.warning(
            f"    Usando unión posicional parcial: "
            f"primeras {n_admet} filas de {n_ref}"
        )
        admet_df.index = base_df.index[:n_admet]
        joined = base_df.join(admet_df, rsuffix=suffix)
        return joined

    for df, suffix in [(df_sw, "_sw"), (df_pk, "_pk"), (df_al, "_al")]:
        if df is not None:
            base = _join(base, df, suffix)

    # ── Criterios consolidados ────────────────────────────────────────────────
    def get(row, *cols):
        for c in cols:
            if c in row.index and row[c] is not None:
                try:
                    v = row[c]
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    return v
                except (TypeError, ValueError):
                    continue
        return None

    results = []
    for _, row in base.iterrows():
        # BBB: voto por mayoría
        bbb_sw = get(row, "BBB_ok_sw")
        bbb_pk = get(row, "BBB_ok_pk")
        bbb_al = get(row, "BBB_ok_al")
        bbb_ok = majority_vote(bbb_sw, bbb_pk, bbb_al)
        logBB = get(row, "BBB_logBB", "BBB_logBB_al")

        # P-gp
        pgp_sw = get(row, "Pgp_ok_sw")
        pgp_pk = get(row, "Pgp_ok_pk")
        pgp_al = get(row, "Pgp_ok_al")
        pgp_ok = majority_vote(pgp_sw, pgp_pk, pgp_al)

        # hERG
        herg_pk = get(row, "hERG_ok_pk")
        herg_al = get(row, "hERG_ok_al")
        herg_ok = majority_vote(herg_pk, herg_al)
        herg_pic50 = get(row, "hERG_pIC50_al")

        # Microsomal
        micro_ok = get(row, "Microsomal_ok_pk")
        t_half = get(row, "t_half_min_pk")

        # DILI
        dili_pk = get(row, "DILI_ok_pk")
        dili_al = get(row, "DILI_ok_al")
        dili_ok = majority_vote(dili_pk, dili_al)
        dili_risk = get(row, "DILI_risk_al", "Hepatotox_pk")

        # AMES
        ames_ok = get(row, "AMES_ok_al")

        # BCRP
        bcrp_ok = get(row, "BCRP_ok_al")

        # CNS MPO (ya calculado con RDKit)
        cns_mpo = get(row, "CNS_MPO")
        cns_ok = (cns_mpo >= 4.0) if cns_mpo is not None else None

        # ADMET_pass: evaluamos los 7 criterios clave (incluyendo DILI y AMES).
        # Los criterios inciertos (None) no hacen fallar el pase (asumidos como válidos
        # para futura validación in vitro/in vivo según petición).
        core = [cns_ok, bbb_ok, pgp_ok, herg_ok, micro_ok, dili_ok, ames_ok]
        known = [c for c in core if c is not None]
        admet_pass = all(known) if known else None
        uncertain = sum(1 for c in core if c is None)

        results.append(
            {
                **{
                    c: row[c]
                    for c in base.columns
                    if c
                    in [
                        "name",
                        "smiles_canon",
                        "smiles",
                        "dock_score",
                        "ifp_similarity",
                        "composite_score",
                        "CNS_MPO",
                    ]
                    and c in row.index
                },
                "CNS_MPO": cns_mpo,
                "CNS_MPO_ok": cns_ok,
                "BBB_logBB": logBB,
                "BBB_ok": bbb_ok,
                "Pgp_ok": pgp_ok,
                "hERG_pIC50": herg_pic50,
                "hERG_ok": herg_ok,
                "t_half_min": t_half,
                "Microsomal_stable": micro_ok,
                "DILI_risk": dili_risk,
                "DILI_ok": dili_ok,
                "AMES_ok": ames_ok,
                "BCRP_ok": bcrp_ok,
                "ADMET_pass": admet_pass,
                "ADMET_uncertain_criteria": uncertain,
                "sources_used": sum(1 for x in [df_sw, df_pk, df_al] if x is not None),
            }
        )

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Priorización multi-criterio
# ─────────────────────────────────────────────────────────────────────────────


def prioritize(df: pd.DataFrame) -> pd.DataFrame:
    """Puntuación de priorización: 0.40·S_dock + 0.35·S_IFP + 0.25·S_CNS"""
    out = df.copy()
    out["name"] = out["name"].astype(str)

    ds = out["dock_score"].values if "dock_score" in out.columns else np.zeros(len(out))
    valid = ~np.isnan(ds.astype(float))
    s_dock = np.zeros(len(out))
    if valid.sum() > 1:
        dmin, dmax = ds[valid].min(), ds[valid].max()
        s_dock[valid] = np.clip((ds[valid] - dmax) / (dmin - dmax + 1e-9), 0, 1)

    ifp = (
        out["ifp_similarity"].fillna(0).values
        if "ifp_similarity" in out.columns
        else np.zeros(len(out))
    )
    cns = (
        out["CNS_MPO"].fillna(0).values
        if "CNS_MPO" in out.columns
        else np.zeros(len(out))
    )

    out["dock_score_norm"] = s_dock
    out["prioritization_score"] = (
        0.40 * s_dock + 0.35 * np.clip(ifp, 0, 1) + 0.25 * np.clip(cns / 6.0, 0, 1)
    )
    return out.sort_values("prioritization_score", ascending=False).reset_index(
        drop=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figura ADMET heatmap
# ─────────────────────────────────────────────────────────────────────────────


def plot_admet_heatmap(df: pd.DataFrame, out_path: str) -> None:
    crit_cols = [
        "CNS_MPO_ok",
        "BBB_ok",
        "Pgp_ok",
        "hERG_ok",
        "Microsomal_stable",
        "DILI_ok",
        "AMES_ok",
    ]
    crit_lbl = ["CNS MPO", "BHE", "P-gp", "hERG", "Microsomal", "DILI", "AMES"]
    avail = [c for c in crit_cols if c in df.columns]
    avail_lbl = [crit_lbl[crit_cols.index(c)] for c in avail]
    top = df.head(15)

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), facecolor="white", gridspec_kw={"width_ratios": [2, 1]}
    )

    # Heatmap
    ax = axes[0]
    hm = top[avail].fillna(0.5).astype(float).values.T
    im = ax.imshow(hm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(
        [f"#{i+1}\n{str(r['name'])[:10]}" for i, r in top.iterrows()], fontsize=7.5
    )
    ax.set_yticks(range(len(avail_lbl)))
    ax.set_yticklabels(avail_lbl, fontsize=10)
    ax.set_title(
        "Criterios ADMET por candidato\n"
        "(verde=cumple, rojo=no cumple, amarillo=incierto)",
        fontweight="bold",
    )
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm[i, j]
            sym = "✓" if v >= 0.9 else "✗" if v <= 0.1 else "?"
            ax.text(
                j,
                i,
                sym,
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Puntuación de priorización
    ax2 = axes[1]
    prio = (
        top["prioritization_score"].values
        if "prioritization_score" in top.columns
        else np.zeros(len(top))
    )
    colors = [
        "#1B5E20" if s >= 0.75 else "#43A047" if s >= 0.65 else "#FF8F00" for s in prio
    ]
    ax2.barh(range(len(top))[::-1], prio, color=colors, edgecolor="white")
    for i, s in enumerate(prio[::-1]):
        ax2.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.axvline(0.65, color="#E91E63", ls="--", lw=2, label="Umbral ≥ 0.65")
    ax2.set_xlabel("Puntuación de priorización")
    ax2.set_title("Priorización\nMulti-criterio", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Figura guardada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Importar resultados ADMET de webs y generar priorización"
    )
    ap.add_argument(
        "--input",
        "-i",
        default="admet_inputs",
        help="Directorio con los CSV descargados de las webs",
    )
    ap.add_argument("--hits", "-H", required=True, help="results/hits_top5pct.csv")
    ap.add_argument("--output", "-o", default="results")
    ap.add_argument("--figures", "-f", default="figures")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig = Path(args.figures)
    fig.mkdir(parents=True, exist_ok=True)

    # Cargar referencia
    df_ref = pd.read_csv(args.hits)
    df_ref["name"] = df_ref["name"].astype(str)
    log.info(f"Candidatos de referencia: {len(df_ref)}")

    # Cargar resultados de cada plataforma (los que existan)
    df_sw, df_pk, df_al = None, None, None

    sw_path = inp / "02_swissadme_results.csv"
    if sw_path.exists():
        log.info(f"\nCargando SwissADME: {sw_path}")
        df_sw = parse_swissadme(str(sw_path))
    else:
        log.warning(f"No encontrado: {sw_path}")

    pk_path = inp / "02_pkcsm_results.csv"
    if pk_path.exists():
        log.info(f"\nCargando pkCSM: {pk_path}")
        df_pk = parse_pkcsm(str(pk_path))
    else:
        log.warning(f"No encontrado: {pk_path}")

    al_path = inp / "02_admetlab_results.csv"
    if al_path.exists():
        log.info(f"\nCargando ADMETlab: {al_path}")
        df_al = parse_admetlab(str(al_path))
    else:
        log.warning(f"No encontrado: {al_path}")

    if all(x is None for x in [df_sw, df_pk, df_al]):
        log.error("\nNo se encontró ningún fichero de resultados ADMET.")
        log.error(f"Coloca los CSV en: {inp}/")
        log.error("Nombres esperados:")
        log.error("  02_swissadme_results.csv")
        log.error("  02_pkcsm_results.csv")
        log.error("  02_admetlab_results.csv")
        return

    n_sources = sum(1 for x in [df_sw, df_pk, df_al] if x is not None)
    log.info(f"\nFuentes disponibles: {n_sources}/3")

    # Consolidar
    log.info("\nConsolidando resultados por voto de mayoría...")
    df_consol = consolidate(df_sw, df_pk, df_al, df_ref)

    # Priorizar
    df_final = prioritize(df_consol)

    # Guardar
    df_final.to_csv(out / "candidates_final_ranked.csv", index=False)
    df_consol.to_csv(out / "admet_results.csv", index=False)

    # Resumen
    n_pass = df_final["ADMET_pass"].fillna(False).sum()
    n_unc = (df_final["ADMET_uncertain_criteria"] > 0).sum()
    log.info(f"\n{'='*55}")
    log.info(f"  ADMET aprobados (todos criterios conocidos): {n_pass}/{len(df_final)}")
    log.info(f"  Con algún criterio incierto:                  {n_unc}/{len(df_final)}")
    log.info(f"\n  Top 10 candidatos priorizados:")
    log.info(f"  {'Rank':<5} {'ID':<20} {'Prio':>6} {'ADMET':>6} {'Inciertos':>10}")
    log.info(f"  {'-'*50}")
    for i, row in df_final.head(10).iterrows():
        unc = int(row.get("ADMET_uncertain_criteria", 0))
        log.info(
            f"  {i+1:<5} {str(row['name']):<20} "
            f"{row['prioritization_score']:>6.3f} "
            f"{'OK' if row.get('ADMET_pass') else '?':>6} "
            f"{unc:>10} criterio(s)"
        )
    log.info(f"{'='*55}")

    # Figura
    plot_admet_heatmap(df_final, str(fig / "fig_admet_dashboard.png"))

    log.info(f"\n✓ Resultados guardados en {out}/candidates_final_ranked.csv")


if __name__ == "__main__":
    main()
