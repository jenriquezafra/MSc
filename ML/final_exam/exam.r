rda_path <- file.path("ML/final_exam/ML-2026.rda")
load(rda_path)

needed_objs <- c("trainData", "trainClass", "testData", "testClass")
missing <- needed_objs[!vapply(needed_objs, exists, logical(1))]

trainX <- as.matrix(trainData)
testX <- as.matrix(testData)
trainY <- factor(trainClass)
testY <- factor(testClass)
