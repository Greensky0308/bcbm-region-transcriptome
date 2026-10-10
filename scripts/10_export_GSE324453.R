#!/usr/bin/env Rscript
## 导出 GSE324453 Seurat(v5) → Python 可读：counts.mtx + 特征/细胞 + meta.csv
suppressMessages({library(SeuratObject); library(Matrix)})

B <- Sys.getenv("BCBM_ROOT", unset = ".")
f <- file.path(B, "data/GSE324453/GSE324453_seurat_filtered_annotated.rds")
outdir <- file.path(B, "out/GSE324453")
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

x <- readRDS(f)
obj <- x[["RNA"]]
layers <- Layers(obj)
cat("layers:", paste(layers, collapse=","), "\n")
lyr <- if ("counts" %in% layers) "counts" else layers[1]
m <- LayerData(obj, layer=lyr)
cat("导出层:", lyr, "| dim:", dim(m), "\n")

writeMM(m, file.path(outdir, "counts.mtx"))
writeLines(rownames(m), file.path(outdir, "features.tsv"))
writeLines(colnames(m), file.path(outdir, "barcodes.tsv"))
write.csv(x@meta.data, file.path(outdir, "meta.csv"), row.names=TRUE)
cat("落盘:", outdir, "\n")
cat("meta 列:", paste(colnames(x@meta.data), collapse=", "), "\n")
