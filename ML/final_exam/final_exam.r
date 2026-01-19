rda_path <- file.path("ML/final_exam/ML-2026.rda")
load(rda_path)

trainX <- as.matrix(trainData)
testX <- as.matrix(testData)
trainY <- factor(trainClass)
testY <- factor(testClass)

# ---------- 1(a) ----------
pca <- prcomp(trainX, center = TRUE, scale. = TRUE)
var_explained <- (pca$sdev^2) / sum(pca$sdev^2)
var_table <- data.frame(
  PC = paste0("PC", seq_along(var_explained)),
  VarExplained = var_explained,
  CumVar = cumsum(var_explained)
)

k_pca <- which(var_table$CumVar >= 0.85)[1]

cat("1(a)\n")
print(var_table)
cat("\nnumber of components for >= 85% variance:", k_pca, "\n\n")

# reduce dimensionality
train_pca <- pca$x[, 1:k_pca, drop = FALSE]
test_pca <- predict(pca, newdata = testX)[, 1:k_pca, drop = FALSE]

# ---------- 1(b) ----------
# LDA
lda_model <- MASS::lda(train_pca, grouping = trainY)
linear_pred <- predict(lda_model, test_pca)$class
linear_method <- "LDA"

# k-NN
knn_k <- 5
knn_pred <- class::knn(train_pca, test_pca, cl = trainY, k = knn_k)

cat("1(b) \nMethods used:\n")
cat("- linear:", linear_method, "\n")
cat("- nonlinear: kNN (k =", knn_k, ")\n\n")

# ---------- 1(c) ----------
lin_err <- mean(linear_pred != testY)
knn_err <- mean(knn_pred != testY)

cat("1(c) \nMisclassification rates on test set:\n")
cat("-", linear_method, ":", round(lin_err, 4), "\n")
cat("- kNN:", round(knn_err, 4), "\n\n")

# ---------- 1(d) ----------
fullX <- rbind(trainX, testX)
fullY <- factor(c(trainY, testY))

# scale to avoid domination by larger scale features
fullX_scaled <- scale(fullX)

km <- kmeans(fullX_scaled, centers = 2, nstart = 50)
clusters <- km$cluster

ct <- table(cluster = clusters, class = fullY)
cluster_to_class <- apply(ct, 1, function(row) names(row)[which.max(row)])
km_pred <- factor(cluster_to_class[clusters], levels = levels(fullY))

km_incorrect <- sum(km_pred != fullY)

cat("1(d) \nk-means incorrect assignments:", km_incorrect)


# ---------- 1(e)  ----------
mutual_knn_graph <- function(X, k) {
  n <- nrow(X)
  # squared Euclidean distance matrix
  d2 <- as.matrix(dist(X))^2
  diag(d2) <- Inf
  knn_idx <- apply(d2, 1, function(row) order(row)[1:k])
  W <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in knn_idx[, i]) {
      if (i %in% knn_idx[, j]) {
        W[i, j] <- 1
        W[j, i] <- 1
      }
    }
  }
  W
}

spectral_cluster <- function(X, k_nn = 5, n_clusters = 2) {
  W <- mutual_knn_graph(X, k_nn)
  deg <- rowSums(W)
  D_inv <- diag(ifelse(deg > 0, 1 / deg, 0))
  L_rw <- diag(nrow(W)) - D_inv %*% W

  eig <- eigen(L_rw, symmetric = FALSE)
  order_idx <- order(Re(eig$values))
  U <- Re(eig$vectors[, order_idx[1:n_clusters], drop = FALSE])

  kmeans(U, centers = n_clusters, nstart = 20)$cluster
}

spec_clusters <- spectral_cluster(fullX_scaled, k_nn = 5, n_clusters = 2)
ct_spec <- table(cluster = spec_clusters, class = fullY)
cluster_to_class_spec <- apply(ct_spec, 1, function(row) names(row)[which.max(row)])
spec_pred <- factor(cluster_to_class_spec[spec_clusters], levels = levels(fullY))

spec_incorrect <- sum(spec_pred != fullY)

cat("\nincorrect assignments:", spec_incorrect)

