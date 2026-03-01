# Multivariate Probability

This project covers fundamental concepts in multivariate probability and statistics, including mean vectors, covariance matrices, and multivariate Gaussian distributions.

---

## Learning Objectives

After completing this project, I should be able to explain:

- Who Carl Friedrich Gauss was
- What a joint (multivariate) distribution is
- What covariance represents
- What a correlation coefficient is
- What a covariance matrix is
- What a multivariate Gaussian distribution is

---

## Key Concepts

### Carl Friedrich Gauss

Carl Friedrich Gauss (1777–1855) was a German mathematician and physicist who made major contributions to number theory, statistics, astronomy, and analysis.  
The Gaussian (normal) distribution is named after him.

---

### Joint / Multivariate Distribution

A joint distribution describes the probability behavior of two or more random variables simultaneously.

Instead of modeling one variable:
```
P(X)
```

We model:
```
P(X, Y)
```

or in higher dimensions:
```
P(X₁, X₂, ..., X_d)
```

---

### Covariance

Covariance measures how two variables vary together.

- Positive → variables increase together
- Negative → one increases while the other decreases
- Zero → no linear relationship

Mathematically:

Cov(X, Y) = E[(X − μx)(Y − μy)]

---

### Correlation Coefficient

The correlation coefficient normalizes covariance:

ρ = Cov(X, Y) / (σx σy)

It ranges between -1 and 1.

---

### Covariance Matrix

For a dataset with d dimensions, the covariance matrix is a d × d matrix:

- Diagonal elements → variances
- Off-diagonal elements → covariances

It is always symmetric.

---

### Multivariate Gaussian Distribution

A multivariate Gaussian distribution is defined by:

- A mean vector μ
- A covariance matrix Σ

It generalizes the normal distribution to multiple dimensions.

---

## Implemented Function

### mean_cov(X)

Calculates:

- Mean vector of shape (1, d)
- Covariance matrix of shape (d, d)

Constraints:
- X must be a 2D numpy.ndarray
- Must contain at least 2 data points
- numpy.cov is not used

---

## Requirements

- Python 3.5
- NumPy 1.15
- Ubuntu 16.04 LTS
- pycodestyle compliant
- Only `import numpy as np` allowed
