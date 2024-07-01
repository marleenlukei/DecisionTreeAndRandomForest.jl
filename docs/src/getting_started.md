
# Getting Started with DecisionTreeAndRandomForest.jl

This guide will introduce you to the basics of using classification trees with the `DecisionTreeAndRandomForest.jl` package.

## What is a Decision Tree?

A decision tree is a machine learning model used for both classification and regression tasks. It uses a tree-like structure where internal nodes represent features, branches represent decision rules, and each leaf node represents an outcome. Decision trees are easy to interpret and visualize, making them popular for many applications.

## What is a Random Forest?

A random forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. This approach improves accuracy and reduces overfitting.

## Basic Example of a Classification Tree

### Step 1: Import the Module

First, import the `DecisionTreeAndRandomForest` module:

```@example 1
using DecisionTreeAndRandomForest
```

### Step 2: Prepare Training Data

Prepare some training data and their respective labels:

```@example 1
# Prepare some training data and their respective labels:
data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
nothing # hide
```

### Step 3: Initialize a Tree

Initialize a classification tree. There are two constructors:
1. One that only takes the data and labels as parameters.
2. Another that also takes values for `max_depth` and `min_samples_split`.

```@example 1
tree = ClassificationTree(data, labels, split_gini)
other_tree = ClassificationTree(3, 2, split_gini, data, labels)
nothing # hide
```

### Step 4: Build the Tree

Build the tree using the `fit` function.

!!! info
    Currently, the `fit` function uses the Gini-Impurity to find the optimal split. In the future, you can provide a custom function by passing it into the `fit` function.

```@example 1
fit(tree)
nothing # hide
```

### Step 5: Print the Tree

To inspect the tree structure, use the `print_tree` function.

!!! warning
    This function is mainly used for debugging purposes and could be removed in future releases.

```@example 1
print_tree(tree)
```

### Step 6: Classify Test Samples

Create some test samples for classification.

```@example 1
# Create some test samples for classification.
test_data = ["dog" 38.0; "human" 38.0]
nothing # hide
```

We expect the output to be `healthy` for the first sample and `sick` for the second one.

### Step 7: Predict Labels

Retrieve the labels assigned to the test samples using the `predict` function.

```@example 1
prediction = predict(tree, test_data)
```

By following these steps, you can create and use a basic classification tree. This example illustrates how decision trees can be applied to simple datasets for classification tasks.

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
tree = ClassificationTree(data, labels, chi_squared_split)
```

This flexibility allows users to tailor the decision tree construction to their specific needs, enhancing the versatility of the `DecisionTreeAndRandomForest.jl` package.
```
```
