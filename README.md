# Brain region as a confounder in breast cancer brain metastasis transcriptomes

Analysis code for the study *"Brain region is an unreported confounder in
transcriptomic comparisons of breast cancer brain metastases"*.

## What the pipeline does

1. **Region axis in bulk data** (`03b`, `03c`) — differential expression between
   cerebellum- and cerebrum-resected metastases in a bulk RNA-seq cohort of 99
   tumours (GSE322943), and the same test after regressing out a fixed
   normal-brain-content score.
2. **Compartment localisation** (`10`, `11`, `12`, `13`) — single-nucleus RNA-seq
   from the same cohort (GSE324453). Per-compartment pseudobulk region tests, a
   directed test of the bulk signature in each compartment, and leave-one-donor-out
   resampling.
3. **Confounding by region imbalance** (`16`) — split-half reproducibility and a
   simulation that draws two groups with a controlled difference in cerebellum
   fraction and counts the genes that become significant.
4. **Match to normal brain** (`23`) — correlation of the tumour region effect with
   the cerebellum-versus-cortex effect in normal brain (GTEx v8 median TPM).
5. **Counter-example** (`03d`, `03h`, `03i`, `24`) — a multi-primary brain-metastasis
   dataset (GSE325196) in which resection site is fully confounded with the primary
   tumour type.

Figures are produced by `20`–`25`; data tables by `30`.

## Data

Public datasets are **not** redistributed here. See `data/README.md` for the
accessions and where to place the downloaded files. Steps `03b`, `03c`, `14` and
`30` also query the GEO sample metadata over the network at run time (plain `curl`;
honours `HTTPS_PROXY` if set), because the resection-site annotation is carried in
the GEO series, not in the expression matrix.

## Layout

```
scripts/     analysis and figure scripts (run order below)
lib/         figure style helpers (setup_style.py, export_figure.py)
resources/   HGNC Ensembl-to-symbol mapping used to label bulk DE output
data/        place downloaded datasets here (git-ignored; see data/README.md)
out/         intermediate analysis products (git-ignored)
figures/     figure output, PDF + TIF (git-ignored)
tables/      table output, xlsx + docx (git-ignored)
```

## Environment

Python 3.11 or newer:

```bash
pip install -r requirements.txt
```

Step `10` additionally needs R with `SeuratObject` and `Matrix`.

Set `BCBM_ROOT` to point at a different project root if the scripts are run from
outside this directory; it defaults to the directory that contains `scripts/`.

## Run order

```bash
# 0. one-off export of the annotated Seurat object to Matrix Market format
Rscript scripts/10_export_GSE324453.R

# 1. analysis products
python3 scripts/03b_brbrain_region.py      # bulk region DE (GSE322943) + symbol map
python3 scripts/03c_region_adjust.py       # adjust for normal-brain content
python3 scripts/11_gse324453_region.py     # snRNA compartment-wise region DE
python3 scripts/12_gse324453_directed.py   # directed signature test per compartment
python3 scripts/13_gse324453_LODO.py       # leave-one-donor-out
python3 scripts/14_prep_fig_data.py        # cache plotting data (bulk and pseudobulk)
python3 scripts/17_prep_snrna_extra.py     # cache markers and cell composition
python3 scripts/16_fig3_confound.py        # region-imbalance simulation

# 2. the multi-primary counter-example
python3 scripts/03d_sc_process.py          # GSE325196 -> processed .h5ad
python3 scripts/03h_reannotate_v2.py       # cluster re-annotation evidence
python3 scripts/03i_host_region.py         # host-neuron region test after removing SCLC

# 3. figures (PDF + TIF)
for f in 20_fig1 21_fig2 22_fig3 23_fig4 24_fig5 25_suppS1; do python3 scripts/$f.py; done

# 4. tables (xlsx + docx)
python3 scripts/30_export_tables.py
```

## Outputs

- `figures/` — six figures (five main panels plus one supplementary), each as PDF and TIF.
- `tables/` — dataset summary and supplementary data tables.
- `out/` — intermediate products shared between steps.

## License

MIT (see `LICENSE`).
