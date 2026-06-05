"""
final_filter.py — Módulo 8: Filtrado estricto ADMET y generación de candidatos finales
========================================================================================
Aplica un filtrado estricto sobre candidates_final_ranked.csv (salida de
admet_import_results.py) para seleccionar únicamente los compuestos que
cumplen simultáneamente los 7 criterios ADMET definidos en el §3.8.2 del TFM.

A diferencia de admet_import_results.py, que aplica los criterios de forma
consolidada y con voto por mayoría, este script usa los flags booleanos ya
calculados y almacenados en el CSV de candidatos priorizados. Su propósito
es generar la lista definitiva de candidatos para propuesta de síntesis o
adquisición comercial, junto con una figura 2D de estructuras para incluir
en el documento final del TFM.

Criterios aplicados (columnas booleanas en el CSV de entrada):
    CNS_MPO_ok       — CNS MPO ≥ 4.0 (Wager et al., 2010)
    BBB_ok           — Permeabilidad a la barrera hematoencefálica
    Pgp_ok           — No sustrato de P-glicoproteína
    hERG_ok          — No inhibidor de hERG (riesgo cardiotóxico)
    Microsomal_stable — Estabilidad microsomal aceptable
    DILI_ok          — No hepatotóxico según predicción in silico
    AMES_ok          — No mutagénico en el test de Ames

Los criterios inciertos (NaN, datos no disponibles en ninguna plataforma)
se tratan como válidos (fillna(True)) para evitar falsos negativos en la
priorización computacional. Los candidatos con criterios inciertos están
marcados en ADMET_uncertain_criteria para revisión experimental posterior.

Salidas generadas:
    results/candidates_strict_admet_passed.csv  — Lista final de candidatos
    figures/final_candidates_2d.png             — Figura 2D de estructuras

Uso:
    conda activate tfm_alpha7
    python final_filter.py

    NOTA: El script espera encontrar results/candidates_final_ranked.csv.
    Este archivo se genera como salida de admet_import_results.py.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def main():
    # Cargar los candidatos priorizados generados por admet_import_results.py.
    # Este CSV contiene scores de docking, IFP, CNS MPO y los 7 flags booleanos
    # ADMET consolidados por voto de mayoría entre las tres plataformas web.
    df = pd.read_csv("results/candidates_final_ranked.csv")

    # Definir los 7 criterios ADMET clave que deben cumplirse simultáneamente
    # para que un candidato sea considerado viable para propuesta experimental.
    admet_criteria = [
        "CNS_MPO_ok",
        "BBB_ok",
        "Pgp_ok",
        "hERG_ok",
        "Microsomal_stable",
        "DILI_ok",
        "AMES_ok",
    ]

    # Verificar que las columnas booleanas ADMET están presentes en el CSV.
    # Si faltan, es probable que admet_import_results.py no haya completado
    # correctamente la consolidación. En ese caso, abortar con mensaje informativo.
    missing = [c for c in admet_criteria if c not in df.columns]
    if missing:
        print(f"Advertencia: Faltan columnas en el CSV: {missing}")
        return

    # Filtrar compuestos que cumplan todos los criterios (True o Incierto).
    # Los NaN (criterio incierto por ausencia de datos en todas las plataformas)
    # se rellenan con True para no rechazar compuestos con datos insuficientes.
    # Estos candidatos quedan marcados en ADMET_uncertain_criteria en el CSV
    # de salida para que puedan ser validados experimentalmente (§3.8.3).
    df_passed = df[df[admet_criteria].fillna(True).all(axis=1)].copy()

    print("=" * 70)
    print(f"CANDIDATOS QUE SUPERAN LOS 7 CRITERIOS ADMET ESTRICTOS")
    print("=" * 70)
    print(f"Total que cumplen todo: {len(df_passed)} de {len(df)}")
    print("-" * 70)

    if len(df_passed) > 0:
        # Ordenar por puntuación de priorización (mayor a menor).
        # prioritization_score es la métrica compuesta calculada en
        # admet_import_results.py: 0.40·S_dock + 0.35·S_IFP + 0.25·S_CNS.
        df_passed = df_passed.sort_values(by="prioritization_score", ascending=False)

        # Seleccionar solo las columnas más informativas para la visualización
        # en terminal. El CSV exportado contendrá todas las columnas disponibles.
        cols_to_show = [
            "name",
            "prioritization_score",
            "dock_score",
            "ifp_similarity",
            "CNS_MPO",
        ]

        # Mostrar los resultados en terminal
        print(df_passed[cols_to_show].to_string(index=False))

        # Exportar la lista definitiva de candidatos para la propuesta experimental.
        # Este archivo es leído por docking_ecd_selectivity.py para resaltar
        # los finalistas en la figura de selectividad TMD vs ECD.
        df_passed.to_csv("results/candidates_strict_admet_passed.csv", index=False)
        print("-" * 70)
        print("Resultados guardados en: results/candidates_strict_admet_passed.csv")

        # Generar figura 2D con las estructuras de los candidatos finales.
        # Se intenta primero la columna 'smiles_canon' (SMILES canonicalizado
        # por RDKit en el Módulo 1) y se usa 'smiles' como alternativa si no
        # está disponible. Las moléculas con SMILES inválido se omiten en la figura.
        print("Generando figura 2D de los candidatos...")
        mols = []
        legends = []
        for _, row in df_passed.iterrows():
            smi = row.get("smiles_canon")
            if pd.isna(smi):
                smi = row.get("smiles")
            if pd.isna(smi):
                continue
            
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mols.append(mol)
                legends.append(f"{row['name']}\nScore: {row['prioritization_score']:.3f}")
        
        if mols:
            os.makedirs("figures", exist_ok=True)
            # MolsToGridImage genera una cuadrícula de estructuras 2D de RDKit.
            # returnPNG=False devuelve un objeto PIL Image, que se guarda con .save().
            # Si returnPNG=True devolvería bytes PNG directamente (útil para Jupyter).
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=3,
                subImgSize=(300, 300),
                legends=legends,
                returnPNG=False
            )
            img.save("figures/final_candidates_2d.png")
            print("Figura guardada en: figures/final_candidates_2d.png")
    else:
        # Si ningún compuesto supera los 7 criterios simultáneamente, se muestra
        # el desglose de fallos por criterio para facilitar la revisión del umbral
        # de aceptación o la identificación de criterios demasiado restrictivos.
        # En este caso, considerar relajar el criterio más restrictivo o revisar
        # manualmente los compuestos rechazados por un único criterio incierto.
        print("Ningún candidato cumple estrictamente los 7 criterios simultáneamente.")

        # Desglose de fallos por criterio para diagnóstico
        print("\nDesglose de fallos por criterio:")
        for crit in admet_criteria:
            fallan = (~df[crit].fillna(False)).sum()
            print(f"  - No cumplen {crit}: {fallan} compuestos")


if __name__ == "__main__":
    main()
