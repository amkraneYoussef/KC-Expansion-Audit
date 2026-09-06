# Knowledge component expansion in knowledge tracing under aligned training and evaluation

Supporting materials for the manuscript *Knowledge component expansion in knowledge tracing under aligned training and evaluation*.

This repository contains experimental notebooks, archived results and manuscript figures. The two JSON artifacts in `results/` contain the recorded results used to calculate the manuscript's numerical summaries and confidence intervals.

The study examines how KC expansion, duplicate interaction–KC records and training–evaluation alignment affect knowledge-tracing comparisons. It builds on established work on label leakage.

## Executable notebooks

The experiments were run on Kaggle with GPU acceleration.

| Dataset | Kaggle notebook |
|---|---|
| ASSIST2009 and ASSIST2017 structural check | [Open on Kaggle](https://www.kaggle.com/code/youssefamk/assist2009-kc-expansion) |
| Algebra I 2005–2006 replication | [Open on Kaggle](https://www.kaggle.com/code/youssefamk/algebra-2005-kc-expansion) |

Local notebook copies are provided under `notebooks/`. ASSIST2009 and Algebra2005 are the two main experimental datasets. ASSIST2017 is used only for structural and shortcut diagnostics, not as a third full model-training replication.

## Repository contents

```text
notebooks/
  assist2009-kc-expansion-audit.ipynb
  algebra-2005-kc-expansion-audit.ipynb
results/
  assist2009_results.json                 60 recorded training runs
  algebra2005_replication_results.json    55 recorded training runs
figures/
  figure_1_dataset_structure.png
  figure_2_shortcut_baseline.png
  figure_3_cross_dataset_protocols.png
  figure_4_evaluation_protocol_contrasts.png
  figure_5_order_disruption.png
  figure_6_source_decomposition.png
  figure_7_training_evaluation_alignment.png
```

The artifacts contain 115 recorded training runs in total. Figures 3 and 7 show both main datasets. Figure 2 compares ASSIST2009 with the ASSIST2017 structural check; Algebra2005 shortcut results are reported in the manuscript's Table 2.

## Data

Raw datasets are not redistributed. Obtain them from their providers and follow the applicable access and reuse conditions. SHA-256 values identify the exact source files used in the archived experiments.

| Dataset | File | SHA-256 |
|---|---|---|
| [ASSIST2009](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | `skill_builder_data.csv` | `f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc` |
| [ASSIST2017](https://sites.google.com/view/assistmentsdatamining/dataset) | `anonymized_full_release_competition_dataset.csv` | `b5b366b11d9250af319f3117c5ccb39544fd72379026bfd33514e4bbd047bf73` |
| [Algebra2005](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data) | `algebra_2005_2006_train.txt` | `19528668530108555e344e5bfcc68c762f63a458a32038599d189667e4f10407` |

ASSIST2009 and ASSIST2017 are distributed by the ASSISTments initiative. Algebra I 2005–2006 is a KDD Cup 2010 development dataset. Attach the source files to the appropriate notebook and check its input-path configuration.

To verify a source file on Linux or Kaggle:

```bash
sha256sum skill_builder_data.csv
```

## Experimental design

The notebooks follow a common protocol for model configurations, student-level splitting, interaction-based windows, checkpoint selection and evaluation. Dataset-specific preprocessing accommodates their different schemas. Representation comparisons are paired within each dataset; absolute scores across datasets reflect different prediction tasks.

### Representations

- `QL`: one row per interaction, retaining the lowest-ID KC and discarding the remaining KC annotations. This is a reduced-information baseline, not a representation of the complete KC set.
- `EXP`: clean KC expansion, with one consecutive row per distinct KC of an interaction.
- `SHUF`: an order-disruption probe that moves non-first KC rows to non-adjacent positions. It changes chronology as well as consecutive block structure and is not a pure adjacency intervention.
- `LEGACY`: the filtered raw representation retaining duplicate interaction–KC records. It coincides with `EXP` in the processed Algebra2005 data used here.
- `EXP-AIT`: expanded training with the all-in-one information restriction applied during training as well as evaluation.

### Models and evaluation

DKT, AKT-R and simpleKT are controlled in-house implementations, with shared width and optimisation settings described in the notebooks. They are not presented as exact reproductions of every original implementation.

Evaluation distinguishes row-level scoring, interaction-level aggregation of conventional predictions (`fused`), and exact all-in-one prediction. Under all-in-one, target-interaction response labels are unavailable to that interaction's predictions; earlier interaction responses remain available. Checkpoints are selected using all-in-one interaction-fused validation AUC.

Row-level versus fused scoring changes aggregation and target weighting. Fused versus all-in-one evaluation changes response-label visibility on matched interaction targets. These are sequential protocol contrasts, not independent additive causal effects.

### Seeds, splits and windows

Five seeds (42, 123, 7, 2024 and 31) define 80/10/10 student-level train/validation/test splits, shared across conditions within each dataset. Split identifiers are stored in the artifacts.

Windows are constructed from original interactions before expansion and shared across representations: at most 200 interactions, stride 100 for training and 200 for evaluation, with a shared 384-row budget. The notebooks and stored window audits specify the construction in detail.

Paired contrasts use the five seed-wise differences and 95% t-intervals. These summarise variability across the recorded seeded runs and splits, not population-level uncertainty over students or datasets. An interval spanning zero does not establish equivalence; no equivalence tests were performed.

## Reading the results

The JSON artifacts record provenance, configuration, dataset statistics, student splits, shortcut diagnostics, protocol checks and individual runs. Aligned-training results and their design record are included. Additional diagnostic keys differ between datasets.

Run identifiers follow `DATASET/MODEL/CONDITION/SEED`, for example `A09/simpleKT/QL/42` or `ALG05/DKT/EXP-AIT/123`.

| Run field | Meaning |
|---|---|
| `training_protocol` | Conventional or all-in-one-restricted training |
| `test_row` | Row-level test AUC |
| `test_fused` | Interaction-fused test AUC |
| `test_ai_row`, `test_ai_fused` | Corresponding all-in-one evaluation scores |
| `test_n_rows`, `test_n_kcs`, `test_n_interactions` | Scored target counts |
| `best_val_ai` | Selected checkpoint's all-in-one fused validation AUC |
| `history` | Recorded epoch-level training and validation history |
| `minutes` | Recorded training time |

For example, calculate the ASSIST2009 simpleKT question-level mean and sample standard deviation:

```python
import json
import numpy as np

with open("results/assist2009_results.json", encoding="utf-8") as f:
    results = json.load(f)

seeds = [42, 123, 7, 2024, 31]
values = [
    results["runs"][f"A09/simpleKT/QL/{seed}"]["test_ai_fused"]
    for seed in seeds
]
print(f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
```

## Protocol checks

Executable checks and their outcomes are recorded under `protocol_tests`. On the tested configurations and inputs, they assess:

1. Invariance of an interaction's all-in-one predictions to changes in its response labels.
2. Sensitivity to a preceding interaction's response, providing a positive control that historical responses remain accessible.
3. Agreement between DKT's all-in-one and standard computations on first rows.
4. Exclusion of context-only first interactions from scored targets.

These are implementation checks on the tested cases, not an exhaustive proof covering all possible inputs.

## Reproducing the experiments

1. Obtain the source files and verify their hashes. Attach ASSIST2009 and ASSIST2017 to the ASSIST notebook and Algebra2005 to the Algebra notebook.
2. Read each notebook's environment and input-path configuration before execution.
3. Run cells in order on a GPU runtime, including the aligned-training follow-up cells. The notebooks produce results artifacts and diagnostic figures.
4. Preserve compatible partial results and any checkpoints required by the notebook's resumption logic before ending a session.

Results may vary across hardware and software environments, including through changes in optimisation and checkpoint selection. Recorded environments and seeds support reproducibility but do not guarantee bitwise-identical reruns. The archived JSON artifacts contain the results reported in the manuscript.


## Citation

The manuscript is currently unpublished. Please do not cite it as an accepted journal article.

```bibtex
@unpublished{amkrane_kc_expansion,
  title  = {Knowledge component expansion in knowledge tracing under aligned training and evaluation},
  author = {Amkrane, Youssef and Amounas, Fatima and Azrour, Mourade and Bendaoud, Salma},
  year   = {2026}
}
```
