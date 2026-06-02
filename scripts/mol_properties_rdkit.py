"""
mol_properties_rdkit.py  — Módulo 1 del pipeline
===
Calcula propiedades fisicoquímicas de moléculas usando RDKit y evalúa
si cumplen los criterios de filtrado para fármacos con acción en el SNC,
orientados al sitio TMD del nAChR alpha7.

Importable como módulo por library_builder.py y otros scripts.

Dependencias:
    rdkit >= 2023.09
    pandas >= 1.5
    numpy  >= 1.24

# Uso, rpincipalmente como librería para el scripte library_builder.py:
from mol_properties_rdkit import compute_properties, passes_filters, CNS_MPO_CRITERIA

Uso del script:
# Filtrado manual
python mol_properties_rdkit.py --input compounds.smi --output filtered.csv
# Test
python mol_properties_rdkit.py --test
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# RDKit
try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
except ImportError:
    sys.exit(
        "RDKit no encontrado.\n"
        "Instala con:  conda install -c conda-forge rdkit=2023.09"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Cálculo Dinámico de pKa con RDKit (SMARTS)
# PKA para grupos básicos
BASIC_PKA_SMARTS = {
    "Aliphatic_Amine_Primary": ("[NX3H2;!$(NC=O)]", 10.5),
    "Aliphatic_Amine_Secondary": ("[NX3H1;!$(NC=O)]", 10.0),
    "Aliphatic_Amine_Tertiary": ("[NX3H0;!$(NC=O)]", 9.5),
    "Imidazole": ("c1ncnc1", 7.0),
    "Pyridine": ("c1ncccc1", 5.2),
    "Aniline": ("c1ccccc1[NX3]", 4.6),
}

try:
    COMPILED_PKA_SMARTS = {
        k: (Chem.MolFromSmarts(v[0]), v[1]) for k, v in BASIC_PKA_SMARTS.items()
    }
except NameError:
    COMPILED_PKA_SMARTS = {}


def estimate_basic_pka(mol) -> float:
    # Estima el pKa básico del grupo más fuerte en la molécula.
    max_pka = -10.0
    for name, (pat, pka_val) in COMPILED_PKA_SMARTS.items():
        if pat and mol.HasSubstructMatch(pat):
            if pka_val > max_pka:
                max_pka = pka_val
    if max_pka == -10.0:
        return 7.0
    return max_pka


# Criterios de filtrado
@dataclass(frozen=True)
class FilterCriteria:
    # Criterios de filtrado fisicoquímico para moduladores alostéricos
    # del sitio TMD del nAChR alpha7 (Grønlien et al., 2007; Wager et al., 2010).

    mw_min: float = 250.0
    mw_max: float = 450.0
    clogp_min: float = 1.5
    clogp_max: float = 4.5
    tpsa_max: float = 90.0
    hbd_max: int = 3
    hba_max: int = 7
    rotbonds_max: int = 8
    arom_min: int = 1
    arom_max: int = 4
    remove_pains: bool = True
    remove_brenk: bool = True


# Criterios de filtrado utilizado en todo el pipeline
CNS_MPO_CRITERIA = FilterCriteria()


# Cargamos en la memoria los catálogos de alertas PAINS y BRENK
# para filtrar los compuestos y descartar los que los contengan
def _build_catalog(catalog_type) -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(catalog_type)
    return FilterCatalog.FilterCatalog(params)


_PAINS_CATALOG = _build_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
_BRENK_CATALOG = _build_catalog(FilterCatalogParams.FilterCatalogs.BRENK)


# CNS MPO score — Wager et al. (2010) ACS Chem Neurosci
# El CNS MPO es una puntuación que combina varias propiedades fisicoquímicas
# para predecir si una molécula tiene probabilidad de cruzar la barrera hematoencefálica
# y ser activa en el SNC.


def _desirability(val: float, lo: float, hi: float, reverse: bool = False) -> float:
    """Función de deseabilidad lineal en [lo, hi] → [0, 1]."""
    if reverse:
        if val <= lo:
            return 1.0
        if val >= hi:
            return 0.0
        return (hi - val) / (hi - lo)
    if val >= hi:
        return 1.0
    if val <= lo:
        return 0.0
    return (val - lo) / (hi - lo)


# Calcula la deseabilidad de los 6 parámetros del CNS MPO. Solo los componentes
# con una puntuación de deseabilidad mayor a 4 tiene probabilidades "reales"
# de pasar el filtro


def calc_cns_mpo(
    mw: float, clogp: float, tpsa: float, hbd: int, pka_basic: float = 8.0
) -> float:
    """
    CNS MPO score (0–6). Score ≥ 4 correlaciona con éxito en fase I
    para indicaciones neurológicas (Wager et al., 2010).
    """
    clogd = clogp - 0.5  # aproximación sin pKa experimental

    s_clogp = _desirability(clogp, 3.0, 5.0, reverse=True)
    s_clogd = _desirability(clogd, 2.0, 4.0, reverse=True)

    if tpsa < 40:
        s_tpsa = _desirability(tpsa, 20.0, 40.0)
    elif tpsa > 90:
        s_tpsa = _desirability(tpsa, 90.0, 120.0, reverse=True)
    else:
        s_tpsa = 1.0

    s_hbd = _desirability(hbd, 1.0, 4.0, reverse=True)
    s_mw = _desirability(mw, 360.0, 500.0, reverse=True)

    if pka_basic < 7.5:
        s_pkb = _desirability(pka_basic, 5.0, 7.5)
    elif pka_basic > 10.0:
        s_pkb = _desirability(pka_basic, 10.0, 11.0, reverse=True)
    else:
        s_pkb = 1.0

    return round(min(s_clogp + s_clogd + s_tpsa + s_hbd + s_mw + s_pkb, 6.0), 3)


# Clase para registrar cada molécula


@dataclass
class MolRecord:
    name: str = ""
    smiles_input: str = ""
    smiles_canon: str = ""
    # Propiedades RDKit
    MW: float = 0.0
    clogP: float = 0.0
    TPSA: float = 0.0
    HBD: int = 0
    HBA: int = 0
    RotBonds: int = 0
    ArRings: int = 0
    HeavyAtoms: int = 0
    FractionCSP3: float = 0.0
    CNS_MPO: float = 0.0
    # Alertas
    PAINS: bool = False
    PAINS_desc: str = ""
    BRENK: bool = False
    BRENK_desc: str = ""
    # Criterios individuales
    pass_MW: bool = False
    pass_clogP: bool = False
    pass_TPSA: bool = False
    pass_HBD: bool = False
    pass_HBA: bool = False
    pass_RotBonds: bool = False
    pass_ArRings: bool = False
    pass_PAINS: bool = True
    pass_BRENK: bool = True
    pass_all: bool = False
    fail_reason: str = ""
    parse_error: str = ""


# Función central


def compute_properties(
    smiles: str,
    name: str = "",
    criteria: FilterCriteria = CNS_MPO_CRITERIA,
) -> MolRecord:
    """
    Calcula todas las propiedades fisicoquímicas de una molécula con RDKit.
    Devuelve un MolRecord con propiedades calculadas y resultado de filtrado.

    Uso en otros módulos:
        from mol_properties_rdkit import compute_properties
        rec = compute_properties("Cc1cc(C(F)(F)F)cc(C(=O)Nc2ccc(Cl)c(Cl)c2)c1", "PNU-120596")
    """
    rec = MolRecord(name=name, smiles_input=smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec.parse_error = f"SMILES inválido"
        return rec

    # Construye los valores de las propiedades en cada molécula
    rec.smiles_canon = Chem.MolToSmiles(mol)
    rec.MW = round(Descriptors.ExactMolWt(mol), 3)
    rec.clogP = round(Descriptors.MolLogP(mol), 3)
    rec.TPSA = round(Descriptors.TPSA(mol), 2)
    rec.HBD = Lipinski.NumHDonors(mol)
    rec.HBA = Lipinski.NumHAcceptors(mol)
    rec.RotBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rec.ArRings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rec.HeavyAtoms = mol.GetNumHeavyAtoms()
    rec.FractionCSP3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    pka_dyn = estimate_basic_pka(mol)
    rec.CNS_MPO = calc_cns_mpo(rec.MW, rec.clogP, rec.TPSA, rec.HBD, pka_basic=pka_dyn)

    # PAINS (RDKit FilterCatalog completo — 480+ subestructuras)
    if criteria.remove_pains:
        m = _PAINS_CATALOG.GetMatches(mol)
        if m:
            rec.PAINS = True
            rec.PAINS_desc = "; ".join(x.GetDescription() for x in m)
        rec.pass_PAINS = not rec.PAINS

    # BRENK (grupos reactivos inespecíficos)
    if criteria.remove_brenk:
        m = _BRENK_CATALOG.GetMatches(mol)
        if m:
            rec.BRENK = True
            rec.BRENK_desc = "; ".join(x.GetDescription() for x in m)
        rec.pass_BRENK = not rec.BRENK

    # Criterios fisicoquímicos
    rec.pass_MW = criteria.mw_min <= rec.MW <= criteria.mw_max
    rec.pass_clogP = criteria.clogp_min <= rec.clogP <= criteria.clogp_max
    rec.pass_TPSA = rec.TPSA <= criteria.tpsa_max
    rec.pass_HBD = rec.HBD <= criteria.hbd_max
    rec.pass_HBA = rec.HBA <= criteria.hba_max
    rec.pass_RotBonds = rec.RotBonds <= criteria.rotbonds_max
    rec.pass_ArRings = criteria.arom_min <= rec.ArRings <= criteria.arom_max

    # Aplica los criterios físicoquímicos, solo avanza si cumple todos los criterios
    # y la molécula es quimicamente segura
    phys = all(
        [
            rec.pass_MW,
            rec.pass_clogP,
            rec.pass_TPSA,
            rec.pass_HBD,
            rec.pass_HBA,
            rec.pass_RotBonds,
            rec.pass_ArRings,
        ]
    )
    alerts = rec.pass_PAINS and rec.pass_BRENK
    rec.pass_all = phys and alerts

    # Razón del fallo. Devuelve los motivos de fallo cuando alguno de los criterios no se cumple
    if not rec.pass_all:
        fails = []
        if not rec.pass_MW:
            fails.append(f"MW={rec.MW:.1f} Da")
        if not rec.pass_clogP:
            fails.append(f"clogP={rec.clogP:.2f}")
        if not rec.pass_TPSA:
            fails.append(f"TPSA={rec.TPSA:.1f} Å²")
        if not rec.pass_HBD:
            fails.append(f"HBD={rec.HBD}")
        if not rec.pass_HBA:
            fails.append(f"HBA={rec.HBA}")
        if not rec.pass_RotBonds:
            fails.append(f"RotBonds={rec.RotBonds}")
        if not rec.pass_ArRings:
            fails.append(f"ArRings={rec.ArRings}")
        if not rec.pass_PAINS:
            fails.append(f"PAINS:{rec.PAINS_desc[:50]}")
        if not rec.pass_BRENK:
            fails.append(f"BRENK:{rec.BRENK_desc[:50]}")
        rec.fail_reason = " | ".join(fails)

    return rec


def passes_filters(smiles: str, criteria: FilterCriteria = CNS_MPO_CRITERIA) -> bool:
    """Devuelve True si el SMILES supera todos los filtros."""
    rec = compute_properties(smiles, criteria=criteria)
    return rec.pass_all and not rec.parse_error


def batch_process(
    compounds: list[tuple[str, str]],
    criteria: FilterCriteria = CNS_MPO_CRITERIA,
    verbose: bool = True,
) -> list[MolRecord]:
    """
    Procesa una lista de (SMILES, nombre) y devuelve todos los MolRecord.
    Uso desde library_builder.py:
        from mol_properties_rdkit import batch_process
        records = batch_process([(smi, name), ...])
    """
    records = []
    n = len(compounds)
    n_err = 0
    for i, (smi, name) in enumerate(compounds):
        if verbose and i % 1000 == 0 and i > 0:
            log.info(f"  Procesando {i}/{n} ({i/n*100:.0f}%)...")
        rec = compute_properties(smi, name, criteria)
        if rec.parse_error:
            n_err += 1
        records.append(rec)
    if verbose and n_err:
        log.warning(f"  {n_err} moléculas con SMILES inválido omitidas.")
    return records


# Lectura del archivo SMILES o CSV


def read_input(path: str) -> list[tuple[str, str]]:
    """
    Lee .smi, .smiles o .csv y devuelve lista de (smiles, nombre).
    El .csv debe tener columna 'smiles' y si es posible 'name'/'zinc_id'.
    """
    # Introducimos la ruta del archivo y verificamos que exista
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Si es un archivo SMILES o .txt, lo leemos y devolvemos una lista de tuplas (SMILES, nombre)
    if p.suffix.lower() in (".smi", ".smiles", ".txt"):
        out = []
        with open(p) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                out.append((parts[0], parts[1] if len(parts) > 1 else f"mol_{i+1:05d}"))
        return out

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df.columns = df.columns.str.lower().str.strip()

        # Soporte para CSVs descargados de ZINC-22 SmallWorld
        if "alignment" in df.columns and not any(
            c in df.columns for c in ("smiles", "smi")
        ):
            split_cols = df["alignment"].str.split(n=1, expand=True)
            if split_cols.shape[1] == 2:
                df["smiles"] = split_cols[0]
                df["zinc_id"] = split_cols[1]
            else:
                df["smiles"] = split_cols[0]

        smi_col = next(
            (c for c in df.columns if c in ("smiles", "smi", "canonical_smiles")), None
        )
        if not smi_col:
            raise ValueError(
                f"Columna 'smiles' no encontrada en {path}. "
                f"Columnas: {list(df.columns)}"
            )
        name_col = next(
            (c for c in df.columns if c in ("name", "zinc_id", "id", "compound_id")),
            None,
        )
        out = []
        for i, row in df.iterrows():
            smi = str(row[smi_col]).strip()
            name = str(row[name_col]) if name_col else f"mol_{i+1:05d}"
            out.append((smi, name))
        return out

    raise ValueError(f"Formato no soportado: {p.suffix}")


# Genera un informe de texto con estadísticas del filtrado.


def write_report(records: list[MolRecord], criteria: FilterCriteria) -> str:
    df = pd.DataFrame([asdict(r) for r in records])
    valid = df[df["parse_error"] == ""]
    passed = valid[valid["pass_all"]]

    lines = [
        "=" * 60,
        "INFORME DE FILTRADO — Módulo 1 SBDD α7 nAChR",
        "=" * 60,
        f"  Entrada total:               {len(records):>7}",
        f"  Errores de parseo:           {(df['parse_error']!='').sum():>7}",
        f"  Válidas procesadas:          {len(valid):>7}",
        f"  Superan todos los filtros:   {len(passed):>7}  "
        f"({len(passed)/max(len(valid),1)*100:.1f}%)",
        "",
        "── CASCADA DE FILTRADO ─────────────────────────────────",
    ]
    cur = valid.copy()
    for col, lbl in [
        ("pass_MW", f"MW {criteria.mw_min}–{criteria.mw_max} Da"),
        ("pass_clogP", f"clogP {criteria.clogp_min}–{criteria.clogp_max}"),
        ("pass_TPSA", f"TPSA < {criteria.tpsa_max} Å²"),
        ("pass_HBD", f"HBD ≤ {criteria.hbd_max}"),
        ("pass_HBA", f"HBA ≤ {criteria.hba_max}"),
        ("pass_RotBonds", f"RotBonds ≤ {criteria.rotbonds_max}"),
        ("pass_ArRings", f"ArRings {criteria.arom_min}–{criteria.arom_max}"),
        ("pass_PAINS", "PAINS (RDKit)"),
        ("pass_BRENK", "BRENK/REOS"),
    ]:
        nb = len(cur)
        cur = cur[cur[col]]
        lines.append(
            f"  {lbl:<30} eliminados:{nb-len(cur):>5}  retenidos:{len(cur):>5}"
        )

    if len(passed):
        lines += [
            "",
            "── ESTADÍSTICAS BIBLIOTECA FILTRADA ────────────────────",
            f"  {'Propiedad':<18} {'Media':>7} {'Mediana':>7} "
            f"{'P5':>7} {'P95':>7} {'σ':>7}",
        ]
        for col in [
            "MW",
            "clogP",
            "TPSA",
            "HBD",
            "HBA",
            "RotBonds",
            "ArRings",
            "CNS_MPO",
        ]:
            v = passed[col]
            lines.append(
                f"  {col:<18} {v.mean():>7.2f} {v.median():>7.2f} "
                f"{v.quantile(0.05):>7.2f} {v.quantile(0.95):>7.2f} {v.std():>7.2f}"
            )
        n4 = (passed["CNS_MPO"] >= 4).sum()
        lines.append(f"\n  CNS MPO ≥ 4: {n4}/{len(passed)} ({n4/len(passed)*100:.1f}%)")

    lines.append("=" * 60)
    return "\n".join(lines)


# Compuestos para el test (PAMs de referencia + controles)

TEST_COMPOUNDS = [
    # PAMs nAChR alpha7 de referencia
    ("CC1=CC(=NO1)NC(=O)NC2=CC(=C(C=C2OC)OC)Cl", "PNU-120596"),
    ("C1C=CC2C1C(NC3=C2C=C(C=C3)S(=O)(=O)N)C4=CC=CC5=CC=CC=C54", "TQS"),
    (
        "C1C=C[C@H]2[C@@H]1[C@H](NC3=C2C=C(C=C3)S(=O)(=O)N)C4=CC=C(C=C4)Br",
        "GAT107",
    ),
    ("CC1=NOC(=C1)C(=CNC2=CC=C(C=C2)Cl)C(=O)NC3=CC=C(C=C3)Cl", "CCMI"),
    ("C1C=CC2C1C(NC3=C2C=C(C=C3)S(=O)(=O)N)C4=CC=C(C=C4)Br", "4BP-TQS"),
    # Controles negativos
    ("CCCCCCCCCCCCCCCC", "C16_lineal_FAIL"),
    ("O=C(O)c1ccc(N)cc1", "acid_aminobenz_FAIL"),
    ("O=C1C=CC(=O)N1c1ccccc1", "maleimida_PAINS"),
    ("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1.c1ccccc1", "terfenilo_clogP_FAIL"),
    ("Cc1ccc(-c2nc3ccccc3o2)cc1", "benzoxazol"),
]


# CLI. En caso de ejecutar el script desde terminal se ejecutaran los comandos en _cli()
# pero si se importa como librería se puede usar directamente las funciones sin imprimirse.


def _cli():
    ap = argparse.ArgumentParser(
        description="Módulo 1 — Propiedades fisicoquímicas (RDKit)"
    )
    ap.add_argument("--input", "-i", help="Fichero .smi o .csv")
    ap.add_argument("--output", "-o", default="library_filtered.csv")
    ap.add_argument("--all", "-a", default="library_all.csv")
    ap.add_argument("--report", "-r", default="filter_report.txt")
    ap.add_argument(
        "--test", action="store_true", help="Modo test con PAMs de referencia"
    )
    ap.add_argument("--mw-max", type=float, default=450.0)
    ap.add_argument("--mw-min", type=float, default=250.0)
    ap.add_argument("--no-pains", action="store_true")
    ap.add_argument("--no-brenk", action="store_true")
    args = ap.parse_args()

    criteria = FilterCriteria(
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        remove_pains=not args.no_pains,
        remove_brenk=not args.no_brenk,
    )

    if args.test:
        log.info(f"Modo test: {len(TEST_COMPOUNDS)} compuestos de referencia")
        compounds = [(smi, name) for smi, name in TEST_COMPOUNDS]
    elif args.input:
        compounds = read_input(args.input)
        log.info(f"Leídas {len(compounds)} moléculas desde {args.input}")
    else:
        ap.error("Especifica --input o --test")

    log.info("Calculando propiedades con RDKit...")
    records = batch_process(compounds, criteria)

    df_all = pd.DataFrame([asdict(r) for r in records])
    df_pass = df_all[df_all["pass_all"] & (df_all["parse_error"] == "")].sort_values(
        "CNS_MPO", ascending=False
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_pass.to_csv(args.output, index=False)
    df_all.to_csv(args.all, index=False)

    report = write_report(records, criteria)
    Path(args.report).write_text(report)

    print(report)

    if args.test:
        print("\n── TABLA DE RESULTADOS (modo test) ────────────────────")
        cols = [
            "name",
            "MW",
            "clogP",
            "TPSA",
            "HBD",
            "HBA",
            "ArRings",
            "CNS_MPO",
            "pass_all",
            "fail_reason",
        ]
        with pd.option_context("display.max_colwidth", 50, "display.width", 130):
            print(df_all[cols].to_string(index=False))

    log.info(f"Salida: {args.output}  ({len(df_pass)} compuestos filtrados)")


if __name__ == "__main__":
    _cli()
