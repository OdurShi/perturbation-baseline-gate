# Data

This folder holds the small derived files that every result and test in the repository
is computed from. The raw single-cell file is not stored here.

| File | Content |
|---|---|
| `norman2019_pseudobulk.npz` | mean log-normalised expression per perturbation and for control cells, cell counts, differentially expressed gene ranks, reference scores, metadata |
| `norman2019_conditions.csv` | one row per perturbation: number of perturbed genes, number of cells, split-half replicate scores, summed squared standard errors |
| `SHA256SUMS` | SHA-256 of the `.npz` file and of its array contents |

## Source

| | |
|---|---|
| Experiment | Norman et al. 2019, Perturb-seq, K562 cells, CRISPR activation, single and double gene perturbations |
| Release used | scPerturb, Zenodo record 13350497, version 1.4, <https://zenodo.org/records/13350497> |
| File | `NormanWeissman2019_filtered.h5ad` |
| Download URL | <https://zenodo.org/api/records/13350497/files/NormanWeissman2019_filtered.h5ad/content> |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the Zenodo record |
| Size | 698,680,199 bytes |
| MD5 | `c870e6967d91c017d9da827bab183cd6` (matches the value listed on the Zenodo record) |
| SHA-256 | `efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc` (computed after download) |
| Content | 111,445 cells x 33,694 genes, integer UMI counts |

`scripts/download_data.py` downloads the file into `data/raw/` (git-ignored) or into the
folder named by the environment variable `PBGATE_CACHE`, and verifies size, MD5 and
SHA-256. The derived files are redistributed under the same licence, CC BY 4.0; see the
acknowledgement in the top-level README.

## Processing

`scripts/preprocess_pseudobulk.py` performs the following steps.

1. **Cells.** Keep cells that the source metadata flags `good_coverage` and
   `number_of_cells == 1` (one cell per droplet, guide pair assigned with good coverage):
   91,168 of 111,445 cells.
2. **Conditions.** The control is the set of cells labelled `control` (two non-targeting
   guides): 9,702 cells. Perturbations need at least 30 cells; none is dropped. This gives
   105 single and 131 double perturbations (81,466 cells; 46 to 1,629 cells per
   perturbation, median 281). Both single genes of every double are measured.
3. **Names.** Perturbation names are the perturbed genes, sorted and joined with `+`
   (`KLF1`, `CBL+CNN1`). Three outdated symbols are replaced by the symbol of the same
   gene on the gene axis, after a check against HGNC (<https://rest.genenames.org>):
   `C19orf26` -> `CBARP`, `C3orf72` -> `FOXL2NB`, `KIAA1804` -> `MAP3K21`. The count
   matrix lists `MAP3K21` (ENSG00000143674) under the clone name `RP5-862P8.2`; the gene
   axis is relabelled.
4. **Normalisation.** Counts of each cell are divided by the cell's total count over all
   33,694 genes, multiplied by 10,000, and transformed with log(1 + x) (natural log).
5. **Genes.** Keep genes detected (count > 0) in at least 10% of the kept cells (7,640
   genes) plus the 53 perturbed genes below that threshold: 7,693 genes. The filter uses
   all kept cells and is the same for every split.
6. **Pseudobulk.** Mean log-normalised expression per perturbation and for control cells,
   rounded to four decimals and stored as 32-bit floats.
7. **Differentially expressed genes.** For every perturbation, genes are ranked by the
   absolute Welch t statistic between its cells and the control cells (log-normalised
   values). The first 50 gene indices are stored; the metrics use the first 20 after
   removing the perturbed genes themselves.
8. **Split-half replicate.** The cells of each perturbation are split at random into two
   halves (seed 0). The pseudobulk of one half is scored as a prediction of the other half
   with the four metrics of the package. These scores are a reference for the sampling
   noise at half the cell number, not an upper bound.
9. **Sampling noise.** For each perturbation the squared standard errors of the mean
   (variance over cells divided by the number of cells) are summed over the metric gene
   sets (`sem2_all`, `sem2_top1000`); the control values are stored in the metadata.

On-target check written to the metadata: the perturbed gene is higher than in control
cells in 103 of 105 single perturbations and is among the 20 top-ranked genes in 69.

## Layout of `norman2019_pseudobulk.npz`

| Array | Shape | Meaning |
|---|---|---|
| `genes` | (7693,) | gene symbols |
| `conditions` | (236,) | perturbation names, control excluded |
| `mean` | (236, 7693) | mean log-normalised expression |
| `n_cells` | (236,) | cells per perturbation |
| `control_mean` | (7693,) | mean log-normalised expression of control cells |
| `n_control_cells` | scalar | 9,702 |
| `de_genes` | (236, 50) | gene indices ranked by decreasing absolute Welch t |
| `extra_splithalf_<metric>` | (236,) | split-half replicate score per metric |
| `extra_sem2_all`, `extra_sem2_top1000` | (236,) | summed squared standard errors |
| `meta_json` | scalar string | source, parameters and counts of the processing |

Load it with `pbgate.data.load_pseudobulk()` or with `numpy.load`.

## Checksums

`SHA256SUMS` lists the SHA-256 of the `.npz` file and a content hash over the arrays
(`pbgate.data.content_sha256`). The file hash depends on the zlib build that compressed
the archive; the content hash does not. `tests/test_data.py` checks both.

```
d52d54b1a926e791228397fdc43a2850c00c61b48a0235fdc46ed8e6cac0bad9  norman2019_pseudobulk.npz
eb934441d80445eb562483a362313d8c9fcfa2b988f7548291cdf382e8c0ae1b  norman2019_pseudobulk.npz:content
```
