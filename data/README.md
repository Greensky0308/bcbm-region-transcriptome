# Data

Datasets are public and are not redistributed with this repository. Download them
from the accessions below and place them at the paths shown. All GEO series are
open access.

| Dataset | Accession | Used by | Place at | Notes |
|---|---|---|---|---|
| Bulk RNA-seq, 99 breast cancer brain metastases | [GSE322943](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE322943) | `03b`, `03c`, `14`, `25`, `30` | `data/GSE322943/counts.xlsx` | Count matrix; resection site (cerebrum / cerebellum), batch and age are in the GEO sample metadata, fetched at run time |
| Single-nucleus RNA-seq, 28 samples, same cohort | [GSE324453](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE324453) | `10`, `11`, `12`, `13`, `14`, `17`, `25`, `30` | `data/GSE324453/GSE324453_seurat_filtered_annotated.rds` | Annotated Seurat v5 object; GEO also hosts it under superseries GSE324454 together with GSE322943 |
| Multi-primary brain-metastasis snRNA-seq and spatial | [GSE325196](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325196) | `03d`, `03h`, `03i`, `24` | `data/GSE325196/RAW.tar` (extract to `data/GSE325196/`) | Counter-example cohort; mixes five primary tumour types |
| GTEx v8 brain, median TPM per tissue | [GTEx v8](https://gtexportal.org/home/datasets) | `23` | `data/GTEx/gtex_median_tpm.gct.gz` | `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz` |

`resources/hgnc_ensembl_symbol.tsv.gz` is a two-column extract of the
[HGNC](https://www.genenames.org/) gene set (`ensembl_gene_id`, `symbol`) used to
label Ensembl identifiers in the bulk differential-expression output.
