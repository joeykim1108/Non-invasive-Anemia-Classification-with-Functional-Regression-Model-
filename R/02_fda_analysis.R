#
# SCRIPT: 02_fda_analysis.R
#
# PURPOSE: This script performs Functional Data Analysis (FDA) on the processed
#          image data. It includes functions for:
#          - Performing Functional Principal Component Analysis (FPCA).
#          - Training a logistic regression model on the FPCA scores.
#          - Evaluating the model's performance using a train/test split.
#          - Performing cross-validation.
#
# AUTHOR: [Your Name]
# DATE: [Date of Last Edit]
#

# =============================================================================
#  1. Load Libraries
# =============================================================================
# It's best practice to load all required packages at the top of the script.

# For FDA
library("fda")
library("funData")
library("MFPCA")

# For modeling and evaluation
library("caret")
library("pROC")
library("DescTools")

# For plotting and data manipulation
library("plotly")
library("dplyr")


# =============================================================================
#  2. Load Processed Data
# =============================================================================
# Load the .RData file created by the '01_data_processing.R' script.

# load("data/processed_hsi_data.RData")

# NOTE: Using the same placeholder data from Script 1 for this example.
num_images <- 221
data.51.R <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.G <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.B <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.hue <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 360), 51, 51))
data.51.saturation <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.intensity <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
rgbhsi <- data.frame(
  HGb = runif(num_images, 9, 15),
  brightness = runif(num_images),
  shutterspeed = runif(num_images),
  exposure = runif(num_images),
  diagnosis = sample(0:1, num_images, replace = TRUE)
)


# =============================================================================
#  3. Main Analysis Function
# =============================================================================

#' Run the complete FDA to Logistic Regression workflow.
#'
#' @param color_space A string, either "RGB" or "HSI".
#' @param num_pcs A numeric vector of length 3, specifying the number of
#'        principal components to retain for each of the three channels.
#' @param train_ratio The proportion of data to use for the training set.
#' @param seed A random seed for reproducibility.
#' @return A list containing the model summary, confusion matrix, ROC curve object,
#'         and accuracy.

run_fda_workflow <- function(color_space = "HSI", num_pcs = c(1, 2, 1), seed = 1) {
  
  # --- 3.1. Select and Prepare Data ---
  if (color_space == "RGB") {
    pix_list <- list(data.51.R, data.51.G, data.51.B)
    pc_names <- paste0("PC", 1:sum(num_pcs), "_RGB")
  } else if (color_space == "HSI") {
    pix_list <- list(data.51.hue, data.51.saturation, data.51.intensity)
    pc_names <- paste0("PC", 1:sum(num_pcs), "_HSI")
  } else {
    stop("color_space must be 'RGB' or 'HSI'")
  }
  
  # Convert lists of matrices to 3D arrays
  arrays <- lapply(pix_list, function(p) aperm(array(unlist(p), dim = c(51, 51, length(p))), c(3, 1, 2)))
  
  # --- 3.2. Train/Test Split ---
  set.seed(seed)
  train_indices <- createDataPartition(rgbhsi$diagnosis, p = 0.8, list = FALSE, times = 1)
  
  train_data_arrays <- lapply(arrays, function(a) a[train_indices, , ])
  test_data_arrays <- lapply(arrays, function(a) a[-train_indices, , ])
  
  train_covariates <- rgbhsi[train_indices, ]
  test_covariates <- rgbhsi[-train_indices, ]
  
  # --- 3.3. Convert to Functional Data & Run FPCA ---
  interval <- (1:51 - 1) / 50
  
  # Create funData objects for the training set
  train_fun_data <- lapply(train_data_arrays, function(X) funData(argvals = list(interval, interval), X = X))
  
  # Run FPCA on each channel
  fpca_results <- mapply(function(fd, M) {
    MFPCA(mFData = multiFunData(fd), M = M, uniExpansions = list(list(type = "splines2D", k = c(5, 5))))
  }, train_fun_data, num_pcs, SIMPLIFY = FALSE)
  
  # Extract training scores
  train_scores <- do.call(cbind, lapply(fpca_results, `[[`, "scores"))
  
  # --- 3.4. Calculate Test Scores ---
  
  # Create funData objects for the test set
  test_fun_data <- lapply(test_data_arrays, function(X) funData(argvals = list(interval, interval), X = X))
  
  # Project test data onto the eigenfunctions from the training data
  test_scores <- do.call(cbind, mapply(function(fpca, fun_test) {
    predict(fpca, new_mFData = multiFunData(fun_test), M = fpca$M)
  }, fpca_results, test_fun_data, SIMPLIFY = FALSE))
  
  
  # --- 3.5. Build and Evaluate Logistic Regression Model ---
  
  # Combine scores and covariates into data frames
  train_df <- as.data.frame(train_scores)
  names(train_df) <- pc_names
  train_df <- cbind(train_df, train_covariates)
  
  test_df <- as.data.frame(test_scores)
  names(test_df) <- pc_names
  test_df <- cbind(test_df, test_covariates)
  
  # Create the model formula dynamically
  formula_str <- paste("as.factor(diagnosis) ~", paste(pc_names, collapse = " + "), "+ brightness + shutterspeed + exposure")
  model_formula <- as.formula(formula_str)
  
  # Train the model
  glm_model <- glm(model_formula, data = train_df, family = binomial)
  
  # Make predictions on the test set
  pred_probs <- predict(glm_model, newdata = test_df, type = "response")
  
  # Evaluate
  roc_curve <- roc(test_df$diagnosis, pred_probs, quiet = TRUE)
  best_threshold <- coords(roc_curve, "best", ret = "threshold")$threshold
  pred_class <- as.factor(ifelse(pred_probs >= best_threshold, 1, 0))
  
  # Ensure levels match for confusion matrix
  levels(pred_class) <- c("0", "1")
  test_df$diagnosis <- as.factor(test_df$diagnosis)
  
  conf_matrix <- confusionMatrix(data = pred_class, reference = test_df$diagnosis)
  
  # --- 3.6. Return Results ---
  return(list(
    model = glm_model,
    summary = summary(glm_model),
    confusionMatrix = conf_matrix,
    roc = roc_curve,
    accuracy = conf_matrix$overall['Accuracy']
  ))
}

# =============================================================================
#  4. Example Usage
# =============================================================================

# --- Run the analysis for the HSI color space ---
# Retain 1 PC for Hue, 2 for Saturation, and 1 for Intensity
print("Running FDA workflow for HSI...")
hsi_results <- run_fda_workflow(color_space = "HSI", num_pcs = c(1, 2, 1), seed = 123)

# Print the key results
print(hsi_results$summary)
print(hsi_results$confusionMatrix)
cat("HSI Model Accuracy:", hsi_results$accuracy, "\n")
cat("HSI Model AUC:", auc(hsi_results$roc), "\n")

# Plot the ROC curve
plot(hsi_results$roc, main = "ROC Curve for HSI-based FDA Model")


# --- Run the analysis for the RGB color space ---
# Retain 1 PC for Red, 1 for Green, and 1 for Blue
print("Running FDA workflow for RGB...")
rgb_results <- run_fda_workflow(color_space = "RGB", num_pcs = c(1, 1, 1), seed = 123)

# Print the key results
print(rgb_results$summary)
print(rgb_results$confusionMatrix)
cat("RGB Model Accuracy:", rgb_results$accuracy, "\n")
cat("RGB Model AUC:", auc(rgb_results$roc), "\n")

# Plot the ROC curve
plot(rgb_results$roc, main = "ROC Curve for RGB-based FDA Model")


# =============================================================================
#  5. Cross-Validation (Optional)
# =============================================================================
# The 'caret' package is excellent for this. You can wrap the modeling part
# of the workflow in 'train' for easy cross-validation.

# NOTE: This is a simplified example. A full CV implementation would involve
# re-running the FPCA within each fold, which is computationally intensive.

# First, get the full dataset of FPCA scores (run on all data)
# This is a shortcut; for rigorous validation, FPCA should be in the CV loop.
# ... (code to run FPCA on all data and get scores) ...
# full_df <- cbind(all_scores, rgbhsi)
# 
# train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
# 
# cv_model <- train(diagnosis ~ ., data = full_df,
#                   method = "glm",
#                   family = "binomial",
#                   trControl = train_control,
#                   metric = "ROC")
#
# print(cv_model)