# BEvIS paired Bayes factor analysis of connectivity
#
# Compares connectivity between Baby Einstein (BE) and Inscapes (IS) at every
# ROI pair using paired Bayesian t-tests from the BayesFactor package, then
# plots the resulting log10 Bayes factor matrices.
#
# Input:  connMatrices, a numeric ROI x ROI x scan array in a MATLAB v7.3
#         .mat file. The first half of the scans must be BE and the second
#         half IS, with the same subjects in the same order.
# Output: BF01 and BF10 matrices, heatmaps, and optional CSV files.
#
# Run from the repository root. Update the paths below to match your data.

library(rhdf5)
library(BayesFactor)
library(ggplot2)


# Paths -------------------------------------------------------------------

data_dir    <- "data"
results_dir <- "results"
mat_file    <- file.path(data_dir, "BEvIS_ConnMat_file.mat")


# Load data ---------------------------------------------------------------

# List the file contents to confirm the dataset name
h5ls(mat_file)

M <- h5read(mat_file, "/connMatrices") # .mat file must contain a numeric 3D array named connMatrices

n_roi  <- dim(M)[1]        # number of ROIs
n_pair <- dim(M)[3] / 2    # number of subjects
stopifnot(dim(M)[3] %% 2 == 0)

be_idx <- 1:n_pair                 # BabyEinsteins slices
is_idx <- n_pair + (1:n_pair)      # Inscapes slices

# Row and column indices of each unique ROI pair
edges <- which(upper.tri(matrix(0, n_roi, n_roi)), arr.ind = TRUE)  # ROI*(ROI-1)/2 x 2


# Paired Bayes factors ----------------------------------------------------

# BF10: evidence for a BE vs IS difference
# BF01: evidence for no difference (1 / BF10)
BF01 <- matrix(NA_real_, n_roi, n_roi)
BF10 <- matrix(NA_real_, n_roi, n_roi)

for (e in seq_len(nrow(edges))) {
  i <- edges[e, 1]; j <- edges[e, 2]
  be <- M[i, j, be_idx]
  is <- M[i, j, is_idx]
  ok <- is.finite(be) & is.finite(is)           # drop pairs with NaN in either run

  # Skip edges with too few pairs or no variance in the paired differences
  if (sum(ok) < 3 || sd(be[ok] - is[ok]) == 0) next

  # Paired t-test using the BayesFactor default prior
  bf10 <- extractBF(ttestBF(x = be[ok], y = is[ok], paired = TRUE))$bf
  BF01[i, j] <- 1 / bf10                         # invert: evidence for the null
  BF10[i, j] <- bf10
}

BF01[lower.tri(BF01)] <- t(BF01)[lower.tri(BF01)]  # mirror to full symmetric matrix
BF10[lower.tri(BF10)] <- t(BF10)[lower.tri(BF10)]  # mirror to full symmetric matrix


# Plot --------------------------------------------------------------------

# Convert the matrices to long format for ggplot
df <- expand.grid(i = 1:n_roi, j = 1:n_roi)
df$log10BF01 <- log10(as.vector(BF01))          # column-major, aligns with expand.grid
df$log10BF10 <- log10(as.vector(BF10))

# log10 BF01 heatmap: positive values favour no difference between tasks
ggplot(df, aes(x = j, y = i, fill = log10BF01)) +
  geom_raster() +
  scale_fill_gradient2(
    low = "#2166AC", mid = "grey95", high = "#B2182B",
    midpoint = 0, na.value = "white",
    name = expression(log["10"]~BF["01"])
  ) +
  scale_y_reverse() +                            # drop this line for pcolor-style bottom origin
  coord_fixed() +
  labs(title = "BabyEinsteins vs Inscapes: paired BF",
       x = "ROI", y = "ROI") +
  theme_minimal(base_size = 11)

# Largest absolute value, used to centre the colour scale on 0
L <- max(abs(df$log10BF10), na.rm = TRUE)

# log10 BF10 heatmap: positive values favour a difference between tasks
ggplot(df, aes(x = j, y = i, fill = log10BF10)) +
  geom_raster() +
  scale_fill_gradient2(
    low = "#2166AC", mid = "grey95", high = "#B2182B",
    midpoint = 0, limits = c(-L, L), na.value = "white",
    name = expression(log["10"]~BF["10"])
  ) +
  scale_y_reverse() + coord_fixed() +
  labs(title = "BabyEinsteins vs Inscapes: paired BF", x = "ROI", y = "ROI") +
  theme_minimal(base_size = 11)


# Save results ------------------------------------------------------------

# Uncomment to write the matrices to results_dir
# dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
# write.csv(BF01,        file.path(results_dir, "BEvIS_BF01.csv"),      row.names = FALSE)
# write.csv(BF10,        file.path(results_dir, "BEvIS_BF10.csv"),      row.names = FALSE)
# write.csv(log10(BF01), file.path(results_dir, "BEvIS_log10BF01.csv"), row.names = FALSE)
# write.csv(log10(BF10), file.path(results_dir, "BEvIS_log10BF10.csv"), row.names = FALSE)