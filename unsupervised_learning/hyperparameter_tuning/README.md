# Hyperparameter Optimization (Gaussian Processes)

This project focuses on **hyperparameter tuning** using **Gaussian Processes** and **Bayesian Optimization**. It introduces probabilistic modeling techniques used to efficiently optimize expensive black-box functions such as neural network training.

---

## 📌 Learning Objectives

By the end of this project, you should be able to explain:

### 🔹 Hyperparameter Optimization
- What hyperparameter tuning is
- Difference between grid search and random search
- Why exhaustive search is inefficient

### 🔹 Gaussian Processes
- What a Gaussian Process is
- What a mean function is
- What a kernel function is
- How kernels measure similarity between points

### 🔹 Gaussian Process Regression (Kriging)
- How GP is used for regression
- How predictions include uncertainty (mean + variance)

### 🔹 Bayesian Optimization
- How Bayesian optimization improves search efficiency
- The role of surrogate models (Gaussian Processes)
- Iterative optimization strategy

### 🔹 Acquisition Functions
- Purpose of acquisition functions
- Exploration vs exploitation trade-off

Common acquisition methods:
- Expected Improvement (EI)
- Knowledge Gradient
- Entropy Search / Predictive Entropy Search

---

## ⚙️ Requirements

- Ubuntu 16.04 LTS
- Python 3.5
- NumPy 1.15
- pycodestyle 2.4 (ignore E741)
- All files must be executable
- First line of all files: