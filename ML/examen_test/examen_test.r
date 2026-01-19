# -----------------------------
# 0) Packages
# -----------------------------
pkgs <- c("plsgenomics", "MASS", "e1071", "class", "ggplot2")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(1)  # reproducibility (exam: fix seed)

# -----------------------------
# 1) Load PUBLIC dataset (Colon)
# -----------------------------
data(Colon, package = "plsgenomics")

# to load data use: load("colon.rda")
# Colon is a list with: X (62x2000), Y (length 62, values 1/2)
colon.x <- as.matrix(Colon$X)
colon.y <- as.integer(Colon$Y)

# Create a "test.x" holdout to mimic the exam setup
# (Exam provides test.x externally; here we simulate it.)
n <- nrow(colon.x)
idx_test <- sample(seq_len(n), size = max(10, floor(0.2 * n)))
idx_train <- setdiff(seq_len(n), idx_test)

train.x <- colon.x[idx_train, , drop = FALSE]
train.y <- colon.y[idx_train]
test.x  <- colon.x[idx_test,  , drop = FALSE]
test.y  <- colon.y[idx_test]  # only for evaluation (exam may not give it)

# -----------------------------
# 2) Utilities: scaling + PCA
# -----------------------------
scale_fit <- function(X) list(mu = colMeans(X), sd = apply(X, 2, sd))
scale_apply <- function(X, fit) sweep(sweep(X, 2, fit$mu, "-"), 2, fit$sd, "/")

pca_fit_apply <- function(X_train, X_test = NULL, ncomp = 10) {
  sc <- scale_fit(X_train)
  Xtr <- scale_apply(X_train, sc)
  pca <- prcomp(Xtr, center = FALSE, scale. = FALSE)

  Ztr <- pca$x[, 1:ncomp, drop = FALSE]
  if (is.null(X_test)) return(list(pca = pca, sc = sc, Ztr = Ztr))

  Zte <- predict(pca, newdata = scale_apply(X_test, sc))[, 1:ncomp, drop = FALSE]
  list(pca = pca, sc = sc, Ztr = Ztr, Zte = Zte)
}

best_2_pcs <- function(Z10, y) {
  yfac <- as.integer(as.factor(y))  # 1/2
  score <- sapply(1:ncol(Z10), function(j) {
    a <- Z10[yfac == 1, j]; b <- Z10[yfac == 2, j]
    (mean(a) - mean(b))^2 / (var(a) + var(b) + 1e-12)  # Fisher ratio
  })
  order(score, decreasing = TRUE)[1:2]
}

plot_pca2_with_test <- function(Z2_train, y_train, Z2_test = NULL) {
  df <- data.frame(PC1 = Z2_train[, 1], PC2 = Z2_train[, 2], y = factor(y_train))
  p <- ggplot(df, aes(PC1, PC2, shape = y)) + geom_point(size = 2)
  if (!is.null(Z2_test)) {
    dft <- data.frame(PC1 = Z2_test[, 1], PC2 = Z2_test[, 2])
    p <- p + geom_point(data = dft, aes(PC1, PC2), inherit.aes = FALSE, size = 2)
  }
  p
}

# -----------------------------
# 3) PCA -> 10 comps -> choose 2 best
# -----------------------------
pca_out <- pca_fit_apply(train.x, test.x, ncomp = 10)
Z10_tr <- pca_out$Ztr
Z10_te <- pca_out$Zte

pc2_idx <- best_2_pcs(Z10_tr, train.y)
colon.pca.2 <- Z10_tr[, pc2_idx, drop = FALSE]
test.pca.2  <- Z10_te[, pc2_idx, drop = FALSE]

cat("Selected PC indices (within first 10):", pc2_idx, "\n")

# Plot: train + overlay test
print(plot_pca2_with_test(colon.pca.2, train.y, test.pca.2))

# -----------------------------
# 4) Fisher LDA on colon.pca.2 + predict test
# -----------------------------
lda_fit <- MASS::lda(x = colon.pca.2, grouping = factor(train.y))
prediction <- predict(lda_fit, newdata = test.pca.2)
# prediction$class      predicted class labels
# prediction$posterior  P(class | x) estimated by LDA

cat("\nLDA posterior (first rows):\n")
print(head(prediction$posterior))

# (Optional evaluation if test.y exists)
cat("\nLDA test accuracy (holdout):\n")
print(mean(prediction$class == factor(test.y)))

# -----------------------------
# 5) SVM (RBF) with <= 35 support vectors + posterior probs
# -----------------------------
svm_rbf_maxsv <- function(Xtr2, ytr, maxSV = 35,
                          gammas = 2^(-8:-1), costs = 2^(0:6)) {
  y <- factor(ytr)
  best <- NULL
  for (g in gammas) for (C in costs) {
    m <- e1071::svm(Xtr2, y, kernel = "radial", gamma = g, cost = C, scale = FALSE)
    nsv <- sum(m$nSV)
    if (nsv <= maxSV) {
      acc_tr <- mean(predict(m, Xtr2) == y)
      cand <- list(model = m, gamma = g, cost = C, nSV = nsv, acc = acc_tr)
      if (is.null(best) || cand$acc > best$acc) best <- cand
    }
  }
  best
}

svm_best <- svm_rbf_maxsv(colon.pca.2, train.y, maxSV = 35)

if (is.null(svm_best)) {
  cat("\nNo SVM found with <= 35 SV under this grid. Expanding grid...\n")
  svm_best <- svm_rbf_maxsv(colon.pca.2, train.y, maxSV = 35,
                            gammas = 2^(-10:0), costs = 2^(-1:8))
}

cat("\nChosen SVM hyperparams:\n")
print(list(gamma = svm_best$gamma, cost = svm_best$cost,
           nSV = svm_best$nSV, train_acc = svm_best$acc))

# Refit with probability=TRUE (recommended)
svm_prob <- e1071::svm(colon.pca.2, factor(train.y),
                       kernel = "radial",
                       gamma = svm_best$gamma, cost = svm_best$cost,
                       scale = FALSE, probability = TRUE)

svm_pred_class <- predict(svm_prob, test.pca.2, probability = TRUE)
svm_pred_prob  <- attr(svm_pred_class, "probabilities")

cat("\nSVM prob (first rows):\n")
print(head(svm_pred_prob))
cat("\nSVM test accuracy (holdout):\n")
print(mean(svm_pred_class == factor(test.y)))

# -----------------------------
# 6) k-means on 2000 genes + table(cluster, class)
# -----------------------------
kmeans_eval <- function(X, y, k = 2, nstart = 50) {
  km <- kmeans(X, centers = k, nstart = nstart)
  tab <- table(cluster = km$cluster, class = y)
  list(km = km, tab = tab)
}

km_full <- kmeans_eval(train.x, train.y, k = 2, nstart = 50)
cat("\nKMeans full genes: table(cluster, class)\n")
print(km_full$tab)

# -----------------------------
# 7) k-means with top-10 variance genes
# -----------------------------
top_var_genes <- function(X, p = 10) {
  v <- apply(X, 2, var)
  idx <- order(v, decreasing = TRUE)[1:p]
  list(idx = idx, Xp = X[, idx, drop = FALSE], var = v[idx])
}

tv <- top_var_genes(train.x, p = 10)
km_top10 <- kmeans_eval(tv$Xp, train.y, k = 2, nstart = 50)
cat("\nTop-10 var gene indices:\n")
print(tv$idx)
cat("\nKMeans top-10 var genes: table(cluster, class)\n")
print(km_top10$tab)

# -----------------------------
# 8) kNN on full 2000 genes vs reduced (e.g., PCA2)
# -----------------------------
choose_k_rule <- function(n_train) {
  k <- floor(sqrt(n_train))
  if (k %% 2 == 0) k <- k + 1
  max(k, 3)
}

knn_predict <- function(Xtr, ytr, Xte, k = 5) {
  cl <- class::knn(train = Xtr, test = Xte, cl = factor(ytr), k = k, prob = TRUE)
  p_major <- attr(cl, "prob")  # prob of predicted class
  list(class = cl, prob_pred = p_major)
}

# kNN needs scaling (fit on train)
sc_all <- scale_fit(train.x)
train.x.sc <- scale_apply(train.x, sc_all)
test.x.sc  <- scale_apply(test.x,  sc_all)

k_knn <- choose_k_rule(nrow(train.x))
cat("\nChosen k for kNN:", k_knn, "\n")

knn_full <- knn_predict(train.x.sc, train.y, test.x.sc, k = k_knn)
cat("\nkNN full test accuracy (holdout):\n")
print(mean(knn_full$class == factor(test.y)))
cat("\nkNN full prob (first rows):\n")
print(head(knn_full$prob_pred))

# Reduced version: kNN on PCA2 (already comparable scale, but still ok to scale)
sc_p2 <- scale_fit(colon.pca.2)
p2_tr_sc <- scale_apply(colon.pca.2, sc_p2)
p2_te_sc <- scale_apply(test.pca.2,  sc_p2)

knn_pca2 <- knn_predict(p2_tr_sc, train.y, p2_te_sc, k = k_knn)
cat("\nkNN PCA2 test accuracy (holdout):\n")
print(mean(knn_pca2$class == factor(test.y)))
cat("\nkNN PCA2 prob (first rows):\n")
print(head(knn_pca2$prob_pred))

############################################################
# End.
# Notes:
# - In the real exam, you would remove holdout creation and
#   use the provided test.x (without test.y).
############################################################
