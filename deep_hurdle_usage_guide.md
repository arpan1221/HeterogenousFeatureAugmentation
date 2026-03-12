# Deep Hurdle Networks: Usage Guide and Methodology

## Table of Contents
1. [Quick Start](#quick-start)
2. [Theoretical Background](#theoretical-background)
3. [Detailed Examples](#detailed-examples)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation and Setup

```r
# Source the implementation
source("deep_hurdle_networks.R")

# Run the complete demonstration
results <- run_hurdle_demo()

# The demo will:
# 1. Generate synthetic hurdle data
# 2. Split into train/test
# 3. Train the deep hurdle network
# 4. Evaluate performance
# 5. Create diagnostic visualizations
```

### Basic Usage Pattern

```r
# 1. Initialize the model
model <- DeepHurdleNetwork$new(
  input_dim = 20,                    # Number of features
  stage1_hidden = c(128, 64, 32),    # Participation network architecture
  stage2_hidden = c(128, 64, 32),    # Intensity network architecture
  dropout_rate = 0.2,
  l2_reg = 0.001
)

# 2. Train the model
model$fit(
  X = X_train,
  y1 = y1_train,      # Binary: 1 if participates, 0 otherwise
  y2 = y2_train,      # Continuous: intensity (0 for non-participants)
  epochs = 100,
  batch_size = 32
)

# 3. Make predictions
predictions <- model$predict(X_test, return_components = TRUE)

# predictions contains:
#   $y1_prob     - P(participation)
#   $y2_pred     - E[intensity | participates]
#   $y_expected  - E[y] = P(y1=1) * E[y2|y1=1]
#   $y1_class    - Binary classification

# 4. Evaluate
eval_results <- model$evaluate(X_test, y1_test, y2_test)
```

---

## Theoretical Background

### The Hurdle Model Framework

The hurdle model addresses outcomes with the following structure:

```
y = {  0                           if y₁ = 0
    {  y₂ > 0                       if y₁ = 1
```

Where:
- **y₁**: Binary participation indicator (the "hurdle")
- **y₂**: Continuous positive outcome (conditional on clearing the hurdle)

### Why Standard Models Fail

**Problem 1: Violation of Regression Assumptions**
```r
# WRONG: Standard regression ignores the hierarchical structure
lm(y2 ~ X)  # Violates assumptions - includes structural zeros
```

**Problem 2: Information Loss in Classification**
```r
# WRONG: Binary classification loses intensity information
y_binary <- ifelse(y2 > 0, 1, 0)
glm(y_binary ~ X, family = binomial())  # Throws away "how much"
```

**Problem 3: Deterministic Relationship Not Encoded**
```r
# Standard multi-output models learn: y₂ = 0 when y₁ = 0
# But this is a CONSTRAINT, not a pattern to learn!
```

### The Deep Hurdle Solution

**Stage 1: Participation Model**
```
P(y₁ = 1 | X, Z, W) = σ(f₁(X, Z, W; θ₁))
```
- Binary classification (logistic regression or neural network)
- Handles class imbalance with class weights
- Output: Probability of participation

**Stage 2: Intensity Model**
```
E[y₂ | y₁ = 1, X, Z, W] = f₂(X, Z, W; θ₂)
```
- Continuous regression (linear or neural network)
- Trained ONLY on positive cases (y₁ = 1)
- Output: Expected intensity conditional on participation

**Expected Value**
```
E[y | X, Z, W] = P(y₁ = 1) × E[y₂ | y₁ = 1]
                = σ(f₁(X)) × f₂(X)
```

### Advantages Over Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Standard Regression** | Simple | Violates assumptions; poor on zeros |
| **Binary Classification** | Handles zeros | Loses intensity information |
| **Two-Part OLS** | Econometric standard | Assumes linearity; no interactions |
| **Zero-Inflated Poisson** | Count data models | Requires count outcomes; restrictive |
| **Deep Hurdle** | Nonlinear; flexible; respects structure | More complex; requires more data |

---

## Detailed Examples

### Example 1: Remote Work Prediction

```r
# Your CPS-HPS application
# X: Demographics + workplace features
# y1: Works remotely (yes/no)
# y2: Days per week remote (0-5)

# Load data
cps_data <- read.csv("cps_processed.csv")

# Prepare features
X <- cps_data %>%
  select(age, income, education, occupation, industry, firm_size) %>%
  model.matrix(~ . - 1, data = .)

y1 <- as.numeric(cps_data$remote_work == "remote")
y2 <- cps_data$days_remote  # 0 for non-remote workers

# Train model
model <- DeepHurdleNetwork$new(
  input_dim = ncol(X),
  stage1_hidden = c(256, 128, 64),
  stage2_hidden = c(256, 128, 64),
  dropout_rate = 0.3
)

model$fit(X, y1, y2, epochs = 100)

# Predict for HPS households
hps_data <- read.csv("hps_processed.csv")
X_hps <- prepare_features(hps_data)  # Same encoding as CPS

predictions_hps <- model$predict(X_hps)

# Expected remote work days
hps_data$expected_remote_days <- predictions_hps$y_expected
```

### Example 2: Healthcare Utilization

```r
# Predicting healthcare costs
# y1: Any healthcare utilization (yes/no)
# y2: Total costs (> 0 if y1 = 1)

health_data <- read.csv("healthcare.csv")

# Features: age, chronic conditions, insurance, income
X <- prepare_health_features(health_data)
y1 <- as.numeric(health_data$any_visit > 0)
y2 <- health_data$total_cost

model <- DeepHurdleNetwork$new(
  input_dim = ncol(X),
  stage1_hidden = c(128, 64),
  stage2_hidden = c(128, 64, 32)  # More complex for costs
)

# With importance weights for survey sampling
model$fit(
  X, y1, y2,
  sample_weights = health_data$survey_weight
)
```

### Example 3: Consumer Purchase Behavior

```r
# E-commerce: Purchase amount prediction
# y1: Makes a purchase (yes/no)
# y2: Purchase amount ($)

ecommerce <- read.csv("transactions.csv")

# Features: browsing time, page views, past purchases, etc.
X <- ecommerce %>%
  select(browse_time, page_views, cart_adds, email_opens) %>%
  as.matrix()

y1 <- as.numeric(ecommerce$purchase_amount > 0)
y2 <- ecommerce$purchase_amount

model <- DeepHurdleNetwork$new(
  input_dim = ncol(X),
  stage1_hidden = c(64, 32),    # Simpler for purchase decision
  stage2_hidden = c(128, 64)    # More complex for amount
)

model$fit(X, y1, y2)

# Predict expected revenue for new visitors
new_visitors <- read.csv("new_visitors.csv")
X_new <- prepare_features(new_visitors)
revenue_forecast <- model$predict(X_new)$y_expected
```

---

## Advanced Usage

### 1. Incorporating HFA for Domain Adaptation

```r
# When you have heterogeneous feature spaces
# CPS: workplace features (occupation, industry)
# HPS: household features (family_size, kids)

hfa_model <- HFAHurdleNetwork$new(
  source_dim = 438,    # CPS encoded features
  target_dim = 108,    # HPS encoded features
  common_dim = 150,    # Projection dimension
  hurdle_config = list(
    stage1_hidden = c(256, 128, 64),
    stage2_hidden = c(256, 128, 64)
  )
)

# Train on CPS with labels
hfa_model$fit(
  X_source = X_cps,
  X_target = X_hps,
  y1_source = y1_cps,
  y2_source = y2_cps,
  epochs = 100,
  lambda_mmd = 0.1  # MMD alignment weight
)

# Predict for HPS
predictions_hps <- hfa_model$predict_target(X_hps)
```

### 2. Importance Weighting for Covariate Shift

```r
# When source and target have different distributions

# Step 1: Train discriminator
disc_results <- train_importance_discriminator(
  X_source = X_cps,
  X_target = X_hps
)

cat("ESS ratio:", disc_results$ess_ratio, "\n")
# ESS ratio: 0.563 means good weight concentration

# Step 2: Train with importance weights
model <- DeepHurdleNetwork$new(input_dim = ncol(X_cps))

model$fit(
  X = X_cps,
  y1 = y1_cps,
  y2 = y2_cps,
  sample_weights = disc_results$weights  # Apply importance weights
)

# Now predictions on HPS should be better calibrated
```

### 3. Active Learning for Uncertain Cases

```r
# Identify high-uncertainty predictions for labeling

# Make predictions
predictions <- model$predict(X_hps, return_components = TRUE)

# Compute uncertainty (entropy)
compute_entropy <- function(p) {
  p <- pmax(pmin(p, 0.9999), 0.0001)  # Numerical stability
  -p * log(p) - (1-p) * log(1-p)
}

uncertainty <- compute_entropy(predictions$y1_prob)

# Identify top uncertain cases
uncertain_idx <- order(uncertainty, decreasing = TRUE)[1:500]

# These should be prioritized for:
# 1. Label verification
# 2. Additional feature collection
# 3. Manual review

hps_priority <- hps_data[uncertain_idx, ]
write.csv(hps_priority, "hps_priority_labeling.csv")
```

### 4. Multi-Target Extension

```r
# Predicting multiple outcomes simultaneously
# y1: Remote work (yes/no)
# y2a: Days remote (0-5)
# y2b: Hours remote (0-40)

# Modify Stage 2 to have multiple outputs
build_multitarget_stage2 <- function(input_dim) {
  
  input_layer <- layer_input(shape = input_dim)
  
  # Shared representation
  shared <- input_layer %>%
    layer_dense(128, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(64, activation = "relu")
  
  # Separate output heads
  days_output <- shared %>%
    layer_dense(32, activation = "relu") %>%
    layer_dense(1, activation = "linear", name = "days")
  
  hours_output <- shared %>%
    layer_dense(32, activation = "relu") %>%
    layer_dense(1, activation = "linear", name = "hours")
  
  model <- keras_model(
    inputs = input_layer,
    outputs = list(days_output, hours_output)
  )
  
  model %>% compile(
    optimizer = "adam",
    loss = list(days = "mse", hours = "mse"),
    loss_weights = list(days = 1.0, hours = 0.5)
  )
  
  return(model)
}
```

### 5. Custom Loss Functions

```r
# Focal Loss for severe class imbalance (Stage 1)

focal_loss <- function(alpha = 0.25, gamma = 2.0) {
  function(y_true, y_pred) {
    y_pred <- k_clip(y_pred, k_epsilon(), 1 - k_epsilon())
    
    cross_entropy <- -y_true * k_log(y_pred) - (1 - y_true) * k_log(1 - y_pred)
    
    weight <- y_true * alpha * k_pow(1 - y_pred, gamma) +
              (1 - y_true) * (1 - alpha) * k_pow(y_pred, gamma)
    
    focal <- weight * cross_entropy
    
    return(k_mean(focal))
  }
}

# Modify model compilation
model$stage1_model %>% compile(
  optimizer = "adam",
  loss = focal_loss(alpha = 0.25, gamma = 2.0)
)
```

### 6. Bayesian Uncertainty Quantification

```r
# Monte Carlo Dropout for uncertainty estimation

predict_with_uncertainty <- function(model, X, n_samples = 100) {
  
  # Enable dropout at test time
  predictions_samples <- replicate(n_samples, {
    # This requires modifying the model to keep dropout active
    model$predict(X)
  }, simplify = FALSE)
  
  # Stack predictions
  y1_samples <- do.call(cbind, lapply(predictions_samples, function(p) p$y1_prob))
  y2_samples <- do.call(cbind, lapply(predictions_samples, function(p) p$y2_pred))
  
  # Compute mean and std
  list(
    y1_mean = rowMeans(y1_samples),
    y1_std = apply(y1_samples, 1, sd),
    y2_mean = rowMeans(y2_samples),
    y2_std = apply(y2_samples, 1, sd)
  )
}
```

---

## Troubleshooting

### Issue 1: Stage 2 Won't Train (Insufficient Positives)

**Problem**: "Insufficient positive cases for Stage 2 training"

**Solution**:
```r
# Check positive rate
cat("Positive rate:", mean(y1), "\n")

# If < 10%, consider:
# 1. Collecting more data
# 2. Synthetic oversampling (SMOTE)
# 3. Adjusting classification threshold for Stage 1

library(ROSE)
positive_idx <- which(y1 == 1)
negative_idx <- which(y1 == 0)

# Oversample positives
oversampled <- ovun.sample(
  y1 ~ .,
  data = data.frame(y1, X),
  method = "over",
  N = length(negative_idx) * 2
)$data
```

### Issue 2: Stage 1 Predicting All Zeros or All Ones

**Problem**: Model collapses to predicting majority class

**Solutions**:
```r
# 1. Stronger class weights
class_weights <- list(
  "0" = 1.0,
  "1" = sum(y1 == 0) / sum(y1 == 1)  # Inverse frequency
)

# 2. Focal loss (implemented above)

# 3. SMOTE or other resampling

# 4. Adjust decision threshold
optimal_threshold <- coords(
  roc(y1_test, predictions$y1_prob),
  "best",
  ret = "threshold"
)
y1_class <- ifelse(predictions$y1_prob > optimal_threshold, 1, 0)
```

### Issue 3: Stage 2 Predictions Are Constant

**Problem**: Intensity model predicts same value for everyone

**Solutions**:
```r
# 1. Check for sufficient variation in y2
cat("y2 (positives) summary:\n")
print(summary(y2[y1 == 1]))

# 2. Log-transform highly skewed outcomes
y2_log <- log1p(y2[y1 == 1])  # log(1 + y2)
# Train on log scale, exponentiate predictions

# 3. Add more complex architecture
stage2_hidden = c(256, 256, 128, 64)  # Deeper network

# 4. Reduce regularization
dropout_rate = 0.1  # Less dropout
l2_reg = 0.0001     # Less L2
```

### Issue 4: Overfitting

**Symptoms**: Training loss decreases but validation loss increases

**Solutions**:
```r
# 1. Stronger regularization
model <- DeepHurdleNetwork$new(
  dropout_rate = 0.5,    # Higher dropout
  l2_reg = 0.01          # Stronger L2
)

# 2. Early stopping (already implemented)

# 3. Data augmentation
# Add noise to features during training

# 4. Reduce model complexity
stage1_hidden = c(64, 32)  # Smaller network
```

### Issue 5: Poor Performance on Test Set

**Diagnosis Steps**:
```r
# 1. Check for data leakage
# Are test features from the future?
# Are there duplicate samples?

# 2. Verify feature scaling consistency
# Use same scaler for train and test

# 3. Check domain shift
disc <- train_importance_discriminator(X_train, X_test)
cat("Train-test AUC:", disc$auc, "\n")
# If AUC > 0.7, significant distribution shift

# 4. Evaluate separately by subgroups
for (occupation in unique(test_data$occupation)) {
  idx <- test_data$occupation == occupation
  eval_subset <- model$evaluate(X_test[idx,], y1_test[idx], y2_test[idx])
  cat(occupation, "F1:", eval_subset$stage1$f1, "\n")
}
```

---

## Performance Optimization

### Memory Optimization for Large Datasets

```r
# Use generators for very large data

data_generator <- function(X, y1, y2, batch_size = 32) {
  n_samples <- nrow(X)
  n_batches <- ceiling(n_samples / batch_size)
  
  function() {
    batch_idx <- sample(1:n_samples, batch_size, replace = TRUE)
    list(
      X[batch_idx, , drop = FALSE],
      list(y1[batch_idx], y2[batch_idx])
    )
  }
}

# Use with fit_generator (if implemented)
```

### GPU Acceleration

```r
# Ensure TensorFlow is using GPU
library(tensorflow)
tf$config$list_physical_devices("GPU")

# If available, training will automatically use GPU
# Monitor with:
# nvidia-smi (Linux)
```

### Parallel Training

```r
# Train multiple models in parallel
library(parallel)

cv_folds <- createFolds(y1, k = 5)

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("DeepHurdleNetwork", "X", "y1", "y2"))

cv_results <- parLapply(cl, cv_folds, function(test_idx) {
  train_idx <- setdiff(1:length(y1), test_idx)
  
  model <- DeepHurdleNetwork$new(input_dim = ncol(X))
  model$fit(X[train_idx,], y1[train_idx], y2[train_idx])
  
  model$evaluate(X[test_idx,], y1[test_idx], y2[test_idx])
})

stopCluster(cl)
```

---

## References

### Methodology
- Cragg, J. G. (1971). "Some Statistical Models for Limited Dependent Variables with Application to the Demand for Durable Goods." *Econometrica*, 39(5), 829-844.
- Mullahy, J. (1986). "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3), 341-365.

### Domain Adaptation
- Duan, L., et al. (2012). "Learning with Augmented Features for Heterogeneous Domain Adaptation." *ICML*.
- Huang, J., et al. (2007). "Correcting Sample Selection Bias by Unlabeled Data." *NIPS*.

### Deep Learning
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
- Chollet, F. (2017). *Deep Learning with Python*. Manning.

---

## Support and Contributions

For questions or issues:
1. Check this troubleshooting guide
2. Review the inline code comments
3. Examine the demonstration output from `run_hurdle_demo()`

---

Last Updated: February 2026
