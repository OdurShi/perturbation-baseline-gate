# Notes

Working notes for the results in this repository: exact definitions, assumptions, what
was and was not run, what changed between the first and the second pass, and runtimes.
Every number quoted here is in a file under `results/` or `data/`; the file is named.

## 1. Run record

| | |
|---|---|
| Date of the runs | 14 November 2026 |
| Machine | Apple-silicon laptop (Apple M5, 10 cores, 32 GB memory), macOS, no GPU other than MPS |
| Software | Python 3.12.16, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, matplotlib 3.11.2 (`results/runtime.json`) |
| Second environment for the tests | Python 3.10.22, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3 |
| Offline reproduction (`python scripts/run_all.py`) | 148 s (`results/runtime.json`) |
| Reproduction from raw counts (`python scripts/run_all.py --from-raw`, raw file cached) | 331 s (`results/runtime_from_raw.json`), of which 80 s are the two preprocessing runs |
| Download of the raw file (698,680,199 bytes) | about 5 minutes on the connection used |
| Tests | 60 tests; 14 s with Python 3.12 and 16 s with Python 3.10 on the loaded machine (4 s when idle) |

The machine was running other jobs during the timed runs and the timings varied by a
factor of about three between repetitions on the same day (the run from raw counts took
127 s, 331 s and 422 s in three repetitions; the shipped file is from the last one made
with the final code). The runtimes are upper bounds for an idle machine of this type.
With Python 3.10 (numpy 2.2.6) on this machine the tests emit floating-point warnings
from matrix products; the results agree with the shipped files
(`tests/test_stats_analysis.py`).

## 2. Exact definitions

### Data

- **Source.** scPerturb release of Norman et al. 2019, Zenodo record 13350497 version
  1.4, file `NormanWeissman2019_filtered.h5ad`, CC BY 4.0, 698,680,199 bytes, SHA-256
  `efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc`. The record page
  and its file list (size, MD5) were opened and the MD5 of the download matches.
- **Cells.** 111,445 cells in the file. Kept: `good_coverage` and `number_of_cells == 1`
  (both columns come from the source metadata): 91,168 cells.
- **Control.** The 9,702 kept cells labelled `control` in the source, i.e. cells that
  carry two non-targeting guides (four guide pairs). The same control mean is used by
  the predictors, to define effects, and in the metrics.
- **Perturbations.** 105 single and 131 double perturbations, all with at least 30 cells
  (46 to 1,629, median 281). Each of the 131 doubles has both of its single
  perturbations in the data. Single perturbations delivered with the gene guide in
  either position are one condition, as in the source.
- **Normalisation.** Per cell: counts divided by the cell's total over all 33,694 genes,
  times 10,000, then natural log(1 + x).
- **Genes.** Detected in at least 10% of the kept cells (7,640) plus 53 perturbed genes
  below that threshold: 7,693.
- **Pseudobulk.** Arithmetic mean of the log-normalised values over the cells of a
  condition, rounded to four decimals.
- **Differentially expressed genes.** Welch t statistic per gene between the cells of a
  perturbation and the control cells, on log-normalised values,
  t = (mean_p - mean_c) / sqrt(var_p / n_p + var_c / n_c), ranked by |t|. The top 20
  after removal of the perturbed genes are the perturbation's DE genes. They are
  computed from all cells of the perturbation, including when it is held out; they
  define where a metric looks, and no predictor has access to them.

### Splits

One split per seed (seeds 0 to 19), `pbgate.splits.make_split`:

1. Draw 26 of the 105 perturbed genes (25%, rounded) as unseen genes.
2. Hold out every perturbation that involves an unseen gene.
3. Of the doubles whose two genes are both seen, hold out 25% (rounded) at random.
4. Label held-out perturbations by the realised training set: `unseen_single` (single,
   gene in no training perturbation), `combo_seen0`, `combo_seen1`, `combo_seen2`
   (double with 0, 1, 2 genes that occur in a training perturbation).
5. If a category has fewer than five held-out perturbations, the draw is rejected and
   repeated with the next sub-seed (`rejected_draws` in `results/splits.csv`).

Training sets hold 79 singles and 40 to 63 doubles; held-out sets hold 26 singles, 5 to
17 doubles with no seen gene, 41 to 60 with one and 14 to 21 with two
(`results/splits.csv`). In this dataset every gene that occurs in training also has its
single perturbation in training, so "seen" and "single profile available" coincide; a
test asserts this for seeds 0 to 19.

### Predictors

All are fitted on `TrainData` (gene names, control mean, training profiles) and return
full profiles. Settings are fixed in advance; nothing is tuned on held-out data.

1. `no_change`: the control mean.
2. `train_mean`: unweighted mean of the training perturbations' profiles (singles and
   doubles together).
3. `additive`: control + sum over the perturbed genes of the gene's effect, where the
   effect is (training single profile - control). Fallback for a gene without a
   training single: the mean effect of all training singles. Consequences: a held-out
   single is predicted as control + mean single effect; a double with one seen gene as
   control + that gene's effect + mean single effect; a double with no seen gene as
   control + twice the mean single effect.
4. `linear`: Y ~ base + G W P^T. G (genes x 10): scores of the first 10 principal
   components of the training pseudobulk matrix with genes as observations and training
   perturbations as columns (columns centred). P: for each perturbation the sum of the
   rows of G of its perturbed genes, i.e. the embedding of an unseen perturbation is the
   perturbed gene's own gene embedding, which is computed without that perturbation's
   profile. W (10 x 10): least squares on the training perturbations with ridge penalty
   0.1 on both sides, W = (G'G + 0.1 I)^-1 G' Y_c P (P'P + 0.1 I)^-1. base = control +
   n * m, with n the number of perturbed genes and m the least-squares effect per
   perturbed gene over the training perturbations; with only singles in training this
   equals the mean training profile, the offset b of Ahlmann-Eltze et al. Rank 10 and
   penalty 0.1 are the values given in that paper; the extension to doubles (sum of
   embeddings, offset proportional to n) is ours.
5. `knn`: mean profile of the 5 training perturbations nearest to the query in the same
   embedding space (each dimension scaled to unit variance over genes, Euclidean
   distance).

### Metrics

For a held-out perturbation with predicted profile p, observed profile y and control
mean c, with d_p = p - c and d_y = y - c:

- `pearson_all`: Pearson correlation of d_p and d_y over all genes.
- `pearson_de20`: Pearson correlation over the perturbation's 20 DE genes.
- `mse_de20`: mean of (d_p - d_y)^2 over those 20 genes.
- `l2_top1000`: sqrt(sum of (d_p - d_y)^2) over the 1,000 genes with the highest control
  mean (the definition of Ahlmann-Eltze et al.: L2 distance over the 1,000 most highly
  expressed genes in the control condition).

The perturbed genes themselves are removed from all three gene sets before the sets are
cut to 20 and 1,000 genes. A constant prediction has no defined correlation; it is
reported as 0, which concerns `no_change` only.

### Summaries and intervals

- Mean: over all (seed, perturbation) records of a split category.
- 95% interval: percentile interval of 10,000 bootstrap resamples in which seeds and
  perturbations are resampled independently with replacement and a record is weighted
  by the product of its seed weight and its perturbation weight. This treats the design
  as crossed (the same perturbation is held out under several seeds).
- Pairwise and gate comparisons use the same resampling on paired differences.
- Split-half replicate: for each perturbation, the pseudobulk of a random half of its
  cells is scored against the pseudobulk of the other half. It shows the sampling noise
  at half the cell number. It is not an upper bound: a prediction scored against all
  cells of a perturbation can do better.

### Gate

`pbgate.gate.run_gate`: the model and the five baselines are scored on the same held-out
perturbations of the same split. Advantage = model - baseline for the correlations and
baseline - model for the errors. A comparison is won if the 95% bootstrap interval of
the mean advantage lies above 0.0001 (a margin that prevents rounding in prediction
files from deciding a comparison). A metric passes in a category if all five comparisons
are won; the gate passes if every metric passes in every category with predictions.

## 3. First pass, review, second pass

The first complete pass used: all four gene sets including the perturbed genes; the
linear model with the mean training profile as offset for singles and doubles alike; no
minimum number of perturbations per category. Its numbers are reproduced by the variant
`first_pass` in `results/sensitivity.csv` (they agree with the saved first-pass output to
the printed precision).

What the review checked, and what was done:

| Check | Finding | Action |
|---|---|---|
| Leakage of held-out profiles into predictors | Predictors receive a copy of the control and training rows only. Scrambling the held-out rows leaves every baseline's predictions unchanged (`tests/test_leakage.py`). | none needed; test added |
| Leakage through the split | Unseen genes never occur in training; categories recomputed from the realised training set; A+B and B+A cannot both exist (names are sorted, 131 unique unordered pairs). | validation added to every split |
| Control definition | Control = non-targeting guide pairs, same cell filter. The control mean is shared by prediction and truth in every metric, so its noise is common to both sides. Its summed squared standard error over all genes is 0.080, against a median sum of squared effects of 39.6 per perturbation; the correlation this can induce is about 0.003 (`results/review_checks.json`). | none needed |
| Cell filter | 18% of the cells in the source file are multiplets or have poor guide coverage (`results/review_checks.json`). They were excluded from the start. Without the filter (`results/sensitivity_all_cells.csv`) `pearson_all` changes by at most 0.005 and the errors are lower (more cells per perturbation; `l2_top1000` by 0.30 to 0.60). `additive` stays first on 1/2- and 2/2-seen doubles under every metric; `knn` and `linear`, which are tied in the primary analysis, swap places in three category-metric combinations. | filter kept; sensitivity reported |
| Metric gene sets | The perturbed genes are in the conventional top-20 DE set of 66% of the singles and 63% of the doubles, and carry up to 44% of a perturbation's squared effect over all genes (median 0.5%; `results/review_checks.json`). Under CRISPRa their own up-regulation is the intervention; it is known in advance and trivially additive, and a model could gain on DE metrics by predicting it. | **changed**: perturbed genes removed from all metric gene sets and from the additivity analysis |
| Linear model on doubles | With the mean training profile as offset, a double is predicted as mean + deviation(A) + deviation(B), so the shift shared by all perturbations is counted once although two genes are perturbed. | **changed**: offset = control + n x (effect per perturbed gene); identical to the published form when training has only singles |
| Degenerate splits | The first draw of 6 of the 20 seeds had fewer than five doubles with no seen gene (seeds with `rejected_draws` > 0 in `results/splits.csv`); in the first-pass output one seed had none. | **changed**: such draws are rejected and redrawn |
| Gate decided by rounding | The additive baseline, written to a 32-bit prediction file and read back, was reported as beating itself in two comparisons, with an interval bound that printed as 0.0000. | **changed**: margin of 0.0001 on the interval; tests added |
| Split-half reference used as a leakage alarm | The additive model exceeds the split-half reference on `pearson_all` for doubles with two seen genes (0.862 against 0.818), legitimately, because the reference uses half the cells. | **changed**: the gate warns only when the squared L2 error falls below the sampling-noise floor |

Effect of the changes on the main numbers (`results/sensitivity.csv`, variants
`first_pass` and `primary`; mean over records):

| Category | Predictor | `pearson_all` first -> second | `pearson_de20` | `mse_de20` |
|---|---|---|---|---|
| single, unseen | `train_mean` | 0.501 -> 0.505 | 0.466 -> 0.491 | 0.193 -> 0.174 |
| single, unseen | `additive` | 0.507 -> 0.512 | 0.474 -> 0.499 | 0.202 -> 0.183 |
| single, unseen | `linear` | 0.482 -> 0.475 | 0.497 -> 0.486 | 0.199 -> 0.194 |
| double, 0/2 seen | `linear` | 0.546 -> 0.570 | 0.546 -> 0.575 | 0.358 -> 0.330 |
| double, 1/2 seen | `linear` | 0.634 -> 0.650 | 0.672 -> 0.713 | 0.271 -> 0.218 |
| double, 2/2 seen | `linear` | 0.711 -> 0.719 | 0.774 -> 0.799 | 0.201 -> 0.154 |
| double, 2/2 seen | `additive` | 0.860 -> 0.862 | 0.952 -> 0.951 | 0.046 -> 0.044 |

No conclusion changed between the passes. The largest shifts are in the DE-gene metrics
(removal of the perturbed genes) and in the linear model on doubles (offset). The pooled
R2 of the additive model is 0.678 with and 0.675 without the perturbed genes
(`results/sensitivity_additivity.json`, `results/additivity_summary.json`).

The change to the linear model was made after the first-pass numbers had been seen. It
was made for a structural reason and not selected among alternatives by held-out
performance, but it is a choice informed by held-out results; both variants are in
`results/sensitivity.csv` (`linear_no_gene_shift`).

## 4. Assumptions and limits

- One dataset, one cell line, CRISPR activation, perturbations chosen by the original
  study; nothing here says how the baselines behave on other data.
- Profiles are condition means of log-normalised expression. No cell-level or
  distributional metric is computed.
- The gene set is fixed for all splits and was chosen with all cells (detection rate),
  as is usual; no perturbation labels enter it except to add the perturbed genes.
- Hyperparameters of `linear` (rank 10, ridge 0.1) and `knn` (k = 5) are fixed. The
  sensitivity table shows that they matter: with rank 40 the linear model reaches
  `pearson_all` 0.797 on doubles with two seen genes (0.719 at rank 10) and falls to
  0.447 on unseen singles (0.475 at rank 10); a large ridge penalty turns it into the
  mean-shift prediction (0.501 on unseen singles). No setting brings it above the
  additive model on seen doubles or above the mean on unseen singles.
- The requirement of at least five perturbations per category rejects draws whose unseen
  genes share few doubles; the accepted splits are therefore not a uniform sample of all
  gene subsets (9 rejected draws in 6 of the 20 seeds, `results/splits.csv`).
- The noise correction of the additivity analysis uses standard errors computed from
  the cell-to-cell variance. Batch effects between sequencing lanes are not modelled and
  would count as non-additive signal.
- Intervals describe variation over perturbations and seeds of this dataset. The same
  perturbations recur across seeds, which the crossed bootstrap accounts for; it does
  not account for the dependence between perturbations that share a gene.
- Leakage checks: `check_predictor` can only detect a function that reads the data
  object it is handed. A prediction file produced elsewhere cannot be certified.

## 5. What was not run

- Deep models other than the reduced GEARS run described in section 6; no GPU or MPS
  training was attempted.
- Any dataset other than Norman et al. 2019.
- Hyperparameter selection by inner cross-validation (settings are fixed; see the
  sensitivity table).
- GEO accession GSE133344 was not used as the source: the GEO page could not be opened
  by the automated client (browser verification), so the scPerturb Zenodo record, which
  could be opened and verified, was used instead.
- CI was not executed on a hosted runner; `ruff check .` and `pytest` were run locally
  with Python 3.12 and Python 3.10.

## 6. Optional GEARS run

GEARS (cell-gears 0.1.2) was installed in a separate environment (Python 3.11, torch
2.15.0, torch_geometric 2.8.1, scanpy 1.11.5, anndata 0.12.19, numpy 1.26.4, pandas
2.3.3). With the SciPy version that the installer selected (1.17.1), GEARS failed at model
construction (indexing a sparse matrix with a pandas Series); pinning scipy 1.11.4 fixed
it. No GEARS source file was changed.

`scripts/optional_gears.py` builds a per-cell matrix with the cells, normalisation and
7,693 genes of the pseudobulk file, hands it to GEARS' own data processing, and trains on
the split of seed 0. To fit the time box (45 minutes including installation) the run was
reduced: at most 60 cells per perturbation and 1,000 control cells (15,096 cells), 3
epochs instead of the default 20, batch size 32, hidden size 64, CPU. One tenth of the
training perturbations (13) was set aside for GEARS' model selection, so GEARS trained on
124 perturbations while the baselines use all 141. GEARS drops 7 perturbations whose
genes are not in its gene-ontology graph (FOXL2NB, CBARP, MAP3K21, IER5L and three
doubles containing them); 92 of the 95 held-out perturbations of seed 0 were predicted.
Training took 1,206 s and the whole script 1,379 s, with other jobs running on the
machine. A second run with 6 epochs was started in parallel and stopped after it became
clear that it would not finish inside the time box; no number from it is reported.

Result (`results/optional_gears/gate_seed0.json`): the run fails the gate in every split
category and metric. Over all 92 perturbations `pearson_all` is 0.460 against 0.654 for
the additive baseline, `pearson_de20` 0.551 against 0.730, `mse_de20` 0.165 against
0.141 and `l2_top1000` 4.24 against 3.54. GEARS' own validation error fell in every
epoch (top-20 DE error 0.212, 0.138, 0.127), so the model was not trained to convergence.
This run is evidence that the pipeline accepts a deep model's predictions; it is not
evidence about GEARS at its published training budget, and no published number was
copied into any table.
