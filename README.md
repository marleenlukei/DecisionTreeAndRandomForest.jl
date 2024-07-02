# DecisionTreeAndRandomForest

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/dev/)
[![Build Status](https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/marleenlukei/DecisionTreeAndRandomForest.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/marleenlukei/DecisionTreeAndRandomForest.jl)

DecisionTreeAndRandomForest.jl is a Julia package developed as part of the course project in the module "Julia for Machine Learning" at TU Berlin. It provides an efficient and user-friendly implementation of both decision tree and random forest algorithms. The package supports both classification and regression tasks, leveraging the CART (Classification and Regression Trees) algorithm. Our implementation includes various splitting criteria such as Gini Impurity, Information Gain and Variance Reduction.

## Features

* **Decision Tree:**
    * Supports both classification and regression tasks.
    * Offers a choice of splitting criteria:
        * Gini Impurity
        * Information Gain
        * Variance Reduction 
    * Provides options for controlling tree depth, minimum samples per leaf, and other hyperparameters.
* **Random Forest:**
    * Creates an ensemble of decision trees by randomly sampling features and data points.
    * Improves model robustness and reduces overfitting.
    * Allows for controlling the number of trees, maximum tree depth, and other hyperparameters.

## Getting started
Add the package to your local Julia environment via Pkg by running
```bash
add https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl
```
For further details and examples, please refer to the [docs](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/).