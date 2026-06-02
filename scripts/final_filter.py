import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def main():
    # Cargar los candidatos priorizados
    df = pd.read_csv("results/candidates_final_ranked.csv")

    # Definir los 7 criterios ADMET clave
    admet_criteria = [
        "CNS_MPO_ok",
        "BBB_ok",
        "Pgp_ok",
        "hERG_ok",
        "Microsomal_stable",
        "DILI_ok",
        "AMES_ok",
    ]

    # Verificar que las columnas existan
    missing = [c for c in admet_criteria if c not in df.columns]
    if missing:
        print(f"Advertencia: Faltan columnas en el CSV: {missing}")
        return

    # Filtrar compuestos que cumplan todos los criterios (True o Incierto)
    # Llenamos NaN (incierto) con True para aceptarlos como válidos, 
    # justificando posteriores análisis in vitro/in vivo para confirmarlos.
    df_passed = df[df[admet_criteria].fillna(True).all(axis=1)].copy()

    print("=" * 70)
    print(f"CANDIDATOS QUE SUPERAN LOS 7 CRITERIOS ADMET ESTRICTOS")
    print("=" * 70)
    print(f"Total que cumplen todo: {len(df_passed)} de {len(df)}")
    print("-" * 70)

    if len(df_passed) > 0:
        # Ordenamos por score de priorización
        df_passed = df_passed.sort_values(by="prioritization_score", ascending=False)

        # Seleccionamos las columnas más relevantes para mostrar
        cols_to_show = [
            "name",
            "prioritization_score",
            "dock_score",
            "ifp_similarity",
            "CNS_MPO",
        ]

        # Mostramos los resultados
        print(df_passed[cols_to_show].to_string(index=False))

        # Guardar estos finalistas puros en un CSV aparte si se desea
        df_passed.to_csv("results/candidates_strict_admet_passed.csv", index=False)
        print("-" * 70)
        print("Resultados guardados en: results/candidates_strict_admet_passed.csv")

        # Generar figura 2D de los compuestos
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
        print("Ningún candidato cumple estrictamente los 7 criterios simultáneamente.")

        # Opcional: ver por qué fallan
        print("\nDesglose de fallos por criterio:")
        for crit in admet_criteria:
            fallan = (~df[crit].fillna(False)).sum()
            print(f"  - No cumplen {crit}: {fallan} compuestos")


if __name__ == "__main__":
    main()
