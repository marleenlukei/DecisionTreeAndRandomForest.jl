
# Getting Started with DecisionTreeAndRandomForest.jl

This guide will introduce you to the basics of using classification trees and random forests with the `DecisionTreeAndRandomForest.jl` package.

## What is a Decision Tree?

A decision tree is a machine learning model used for both classification and regression tasks. It uses a tree-like structure where internal nodes represent features, branches represent decision rules, and each leaf node represents an outcome. Decision trees are easy to interpret and visualize, making them popular for many applications.

## What is a Random Forest?

A random forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. This approach improves accuracy and reduces overfitting.

## Overview of Features

- **Classification Trees**: Build trees for classifying data.
- **Regression Trees**: Construct trees for predicting continuous values.
- **Random Forests**: Ensemble method that combines multiple decision trees to improve accuracy and robustness.
- **Custom Splitting Criteria**: Support for various splitting criteria such as Gini Impurity, Information Gain, and Variance Reduction.

## Basic Example of a Classification Tree

### Step 1: Setup

```
julia
using Pkg
Pkg.activate("DecisionTreeAndRandomForest")
Pkg.add(url="https://github.com/marleenlukei/DecisionTreeAndRandomForest.jl/")

```


### Step 2: Import the Module

First, import the `DecisionTreeAndRandomForest` module:

```julia
using DecisionTreeAndRandomForest
```

### Step 3: Prepare Training Data

Prepare some training data and their respective labels:

```julia
data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
```

### Step 4: Initialize a Tree

Initialize a classification tree:

```julia
tree = DecisionTree(-1, 1, -1, split_gini)
```

### Step 5: Build the Tree

Build the tree using the `fit!` function:

```julia
fit!(tree, data, labels)
```

### Step 6: Print the Tree

To inspect the tree structure, simply print the tree:

```julia
print(tree)
```

### Step 7: Classify Test Samples

Create some test samples for classification:

```julia
test_data = ["dog" 38.0; "human" 38.0]
```

We expect the output to be `healthy` for the first sample and `sick` for the second one.

### Step 8: Predict Labels

Retrieve the labels assigned to the test samples using the `predict` function:

```julia
prediction = predict(tree, test_data)
println("Decision Tree Predictions: ", prediction)
```

By following these steps, you can create and use a basic classification tree. This example illustrates how decision trees can be applied to simple datasets for classification tasks.

## Basic Example of a Random Forest

### Step 1: Prepare Training Data

Use the same training data and labels as before:

```julia
data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
```

### Step 2: Initialize a Random Forest

Initialize a random forest with the specified parameters:

```julia
forest = RandomForest(-1, 1, split_gini, 10, 0.8, -1)
```

### Step 3: Build the Random Forest

Build the random forest using the `fit!` function:

```julia
fit!(forest, data, labels)
```

### Step 4: Predict Labels

Create some test samples for classification:

```julia
test_data = ["dog" 38.0; "human" 38.0]
```

Retrieve the labels assigned to the test samples using the `predict` function:

```julia
forest_predictions = predict(forest, test_data)
println("Random Forest Predictions: ", forest_predictions)
```

By following these steps, you can create and use a basic random forest. This example illustrates how random forests can be applied to simple datasets for classification tasks.

## Adding More Splitting Criteria

To add more splitting criteria, define a new function that computes the desired criterion. For example, to implement a Chi-Squared Split:

```julia
using Statistics: chi2

function chi_squared_split(data, labels, num_features)
    # Implementation of Chi-Squared split criterion
end
```

Then use this new function when creating the tree:

```julia
tree = DecisionTree(-1, 1, -1, chi_squared_split)
fit!(tree, data, labels)
```


