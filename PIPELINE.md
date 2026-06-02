# Guía Detallada del Pipeline SBDD α7 nAChR

Esta guía explica paso a paso el funcionamiento, parámetros y comandos del pipeline computacional para identificar moduladores alostéricos positivos del receptor $\alpha7$ nAChR.

---

## Requisitos Previos y Archivos de Entrada

Antes de ejecutar los módulos, asegúrate de tener colocados los archivos de estructura y ligandos de referencia en el directorio `scripts/data/`. El repositorio ya contiene versiones preparadas para la demostración:

*   **Receptor TMD:** `scripts/data/8V82_prepared.pdbqt` (Sitio transmembrana preparado a pH 7.4).
*   **Receptor ECD:** `scripts/data/8V82_ECD_prepared.pdbqt` (Sitio extracelular ortostérico para análisis de selectividad).
*   **Ligando de referencia:** `scripts/data/PNU120596.pdbqt` (PAM de referencia co-cristalizado).
*   **Activos conocidos (benchmarking):** `scripts/data/pam2_actives.smi` (Lista de PAMs conocidos en formato SMILES).
*   **Biblioteca de entrada:** `scripts/data/ZINC_raw_download.csv` (Compuestos descargados de ZINC-22).

---

## Paso 1: Instalación y Validación del Entorno

1.  **Instalar dependencias:** Ejecuta el instalador automático en la raíz del repositorio. Este script instala los paquetes científicos de Conda y Pip, configura la ruta para AutoDock Vina y realiza validaciones iniciales.
    ```bash
    chmod +x setup_environment.sh
    ./setup_environment.sh
    ```
2.  **Activar el entorno conda:**
    ```bash
    conda activate tfm_alpha7
    ```
3.  **Correr verificación de salud:**
    ```bash
    python scripts/verify_installation.py
    ```

---

## Paso 2: Filtrado Fisicoquímico y Preparación de Ligandos

Este paso calcula los descriptores moleculares de las moléculas de entrada utilizando **RDKit**, aplica el score **CNS MPO** con cálculo dinámico de pKa para estimar el paso por la barrera hematoencefálica (BHE) y elimina compuestos con alertas estructurales tóxicas o reactivas (filtros **PAINS** y **BRENK**). También realiza una desduplicación estructural rápida O(N) por similitud de Tanimoto (Morgan Fingerprints).

### Comandos de Ejecución

*   **Modo de prueba (Test):** Ejecuta el filtrado sobre un grupo reducido de compuestos de referencia integrados para validar que los descriptores se calculan correctamente:
    ```bash
    python scripts/library_builder.py --test
    ```

*   **Modo real (Producción):** Filtra una biblioteca descargada (como ZINC-22) y guarda los compuestos que cumplen todos los criterios en `library_filtered.csv`:
    ```bash
    python scripts/library_builder.py \
        --input scripts/data/ZINC_raw_download.csv \
        --output scripts/results/ \
        --figures scripts/figures/
    ```

### Criterios de filtrado aplicados:
*   **Peso Molecular (MW):** $250 \le MW \le 450\text{ Da}$
*   **clogP:** $1.5 \le clogP \le 4.5$
*   **TPSA:** $\le 90\text{ Å}^2$
*   **Donadores de Enlaces de H (HBD):** $\le 3$
*   **Aceptores de Enlaces de H (HBA):** $\le 7$
*   **Enlaces rotables:** $\le 8$
*   **Anillos aromáticos:** $1 \le \text{ArRings} \le 4$
*   **CNS MPO score:** $\ge 4.0$ (correlaciona con éxito en el SNC)

---

## Paso 3: Validación del Protocolo y Cribado Virtual TMD

Este módulo central realiza tres tareas fundamentales:
1.  **Redocking del ligando de referencia (PNU-120596):** Valida la capacidad de AutoDock Vina de reproducir la pose experimental de cry-EM (criterio de éxito: $\text{RMSD} < 2.0\text{ Å}$ en al menos 7 de 10 corridas independientes).
2.  **Benchmarking de enriquecimiento:** Calcula curvas ROC y métricas de enriquecimiento (AUC-ROC $\ge 0.70$, BEDROC $\ge 0.50$ y Factor de Enriquecimiento $EF_{1\%} \ge 5.0$) cruzando los activos conocidos de `pam2_actives.smi` contra decoys generados dinámicamente.
3.  **Cribado prospectivo:** Convierte la biblioteca filtrada en compuestos 3D a pH 7.4 y realiza el docking masivo en paralelo en el sitio TMD. Posteriormente, calcula las huellas de interacción residuo-proteína (**IFP**) usando **ProLIF** y evalúa la similitud de contactos con el PAM de referencia.

### Comando de Ejecución
```bash
python scripts/docking_pipeline.py \
    --receptor scripts/data/8V82_prepared.pdbqt \
    --library scripts/results/library_filtered.csv \
    --actives scripts/data/pam2_actives.smi \
    --ref-ligand scripts/data/PNU120596.pdbqt \
    --output scripts/results/ \
    --figures scripts/figures/
```

### Plan de Contingencia: Recuperación del Docking
Dado que el docking virtual masivo puede tardar varias horas, si la corrida se interrumpe o deseas evaluar los resultados directamente utilizando poses PDBQT ya generadas en el directorio `results/docking_poses/`, puedes ejecutar el recuperador para reconstruir la base de datos de scores y correr el análisis ProLIF de forma instantánea:
```bash
python scripts/recover_from_docking.py \
    --library scripts/results/library_filtered.csv \
    --poses-dir scripts/results/docking_poses/ \
    --receptor-pdb scripts/data/8V82_prepared.pdb \
    --ref-ligand scripts/data/PNU120596.pdbqt \
    --output scripts/results/ \
    --figures scripts/figures/
```

---

## Paso 4: Análisis de Selectividad TMD vs. ECD

Para garantizar que los hits del cribado son selectivos del sitio alostérico (TMD) y no del sitio ortostérico del dominio extracelular (ECD, sitio de unión de la epibatidina), este script realiza un redocking de los candidatos seleccionados contra el ECD y calcula la diferencia de puntuación ($\Delta\text{Score} = \text{Score}_{\text{TMD}} - \text{Score}_{\text{ECD}}$). Buscamos compuestos con valores negativos, indicando una preferencia energética por el TMD.

### Comando de Ejecución
```bash
python scripts/docking_ecd_selectivity.py \
    --screening scripts/results/screening_all.csv \
    --pdbqt-dir scripts/results/pdbqt_library \
    --receptor-ecd scripts/data/8V82_ECD_prepared.pdbqt \
    --output scripts/results/ \
    --figures scripts/figures/
```

---

## Paso 5: Perfil farmacocinético ADMET por Consenso

Las estructuras moleculares del Top 5% filtrado (`hits_top5pct.csv`) se analizan en tres plataformas web externas para la predicción de absorción, distribución, metabolismo, excreción y toxicidad (ADMET):
1.  **SwissADME:** [http://www.swissadme.ch/](http://www.swissadme.ch/)
2.  **pkCSM:** [https://biosig.lab.uq.edu.au/pkcsm/prediction](https://biosig.lab.uq.edu.au/pkcsm/prediction)
3.  **ADMETlab 2.0:** [https://admetlab2.alphama.com.cn/](https://admetlab2.alphama.com.cn/)

### Procedimiento:
1.  Sube los SMILES de `scripts/results/hits_top5pct.csv` a los portales correspondientes.
2.  Descarga los reportes en formato CSV.
3.  Renombra los archivos y colócalos en `scripts/admet_inputs/` con los nombres:
    *   `02_swissadme_results.csv`
    *   `02_pkcsm_results.csv`
    *   `02_admetlab_results.csv`
4.  Ejecuta el importador y consolidador de votación por mayoría:
    ```bash
    python scripts/admet_import_results.py \
        --input scripts/admet_inputs/ \
        --hits scripts/results/hits_top5pct.csv \
        --output scripts/results/ \
        --figures scripts/figures/
    ```

El importador unificará las tablas y resolverá discrepancias mediante **voto por mayoría**, generando `candidates_final_ranked.csv`.

---

## Paso 6: Filtrado Estricto de Candidatos y Dibujo 2D

Aplica los 7 criterios estrictos simultáneos de ADMET para seleccionar a los mejores candidatos finales (hits bioactivos probables y seguros) y dibuja una cuadrícula con sus estructuras moleculares 2D:
```bash
python scripts/final_filter.py
```
*   **Criterios aplicados:** Puntuación CNS MPO $\ge 4.0$, permeabilidad de la BHE por consenso, no sustrato de la Glicoproteína-P (P-gp), no inhibidor de hERG (toxicidad cardíaca), estabilidad microsomal (vida media $> 30\text{ min}$), sin toxicidad hepática (DILI) y Ames negativo (mutagenicidad).
*   **Resultados:** Se almacena el listado en `scripts/results/candidates_strict_admet_passed.csv` y las imágenes en `scripts/figures/final_candidates_2d.png`.
