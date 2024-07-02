# DecisionTreeAndRandomForest

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/dev/)
[![Build Status](https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/marleenlukei/DecisionTreeAndRandomForest.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/marleenlukei/DecisionTreeAndRandomForest.jl)

DecisionTreeAndRandomForest.jl is a Julia package developed as part of the course project in the module "Julia for Machine Learning" at TU Berlin. It provides an efficient and user-friendly implementation of both decision tree and random forest algorithms. The package supports both classification and regression tasks, leveraging the CART (Classification and Regression Trees) algorithm. Our implementation includes various splitting criteria such as Gini Impurity, Information Gain and Variance Reduction.

## Features

### Decision Tree

Decision trees are flowchart-like structures that use a series of "if-then-else" rules to classify or predict an output based on input features. They are created by recursively splitting the data based on the choosen splitting criterion. This package offers the following features for Decision Trees:

* Supports both classification and regression tasks.
* Offers a choice of splitting criteria:
    * Gini Impurity
    * Information Gain
    * Variance Reduction 
* Provides options for controlling tree depth, minimum samples per leaf and other hyperparameters.

### Random Forest

Random forests improve model robustness and reduce overfitting by creating an ensemble of decision trees. Each tree in the ensemble is built using a random subset of the data and features, leading to a diverse set of predictions that are combined for the final result. This package offers the following features for Random Forests:

* Creates an ensemble of decision trees by randomly sampling features and data points.
* Improves model robustness and reduces overfitting.
* Allows for controlling the number of trees, maximum tree depth and other hyperparameters.

## Getting started
Add the package to your local Julia environment via Pkg by running
```bash
add https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl
```
For further details and examples, please refer to the [docs](https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl/).

## Contributing
We welcome contributions from the community. If you would like to contribute to the project, please refer to the [CONTRIBUTING.md] file for more information.
