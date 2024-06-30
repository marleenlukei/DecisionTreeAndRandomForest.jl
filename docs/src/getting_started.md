# Getting Started

This is a basic example on how to use the classification tree.

First, import the module like this

```@example 1
using DecisionTreeAndRandomForest
```

Second we need some training data and their respective labels.

```@example 1
data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
nothing # hide
```

After that we can initialiate a tree. There are two constructors:
1. One only takes the data and labels as parameters.
2. The other one can also take values for `max_depth` and `min_samples_split`.

```@example 1
tree = DecisionTree(split_gini)
other_tree = DecisionTree(3, 2, split_gini)
nothing # hide
```

We can build the tree using the `fit` function.

!!! info
    At the moment the `fit` function uses the Gini-Impurity to find the optimal split. In the future you can provide a custom function by passing it into the `fit` function.


```@example 1
fit!(tree, data, labels)
nothing # hide
```

To take a look at the tree, we can do the following:

```@example 1
print(tree)
```

Lastly, we want to classify some test samples. Therefore we need to create some.

```@example 1
test_data = ["dog" 38.0; "human" 38.0]
nothing # hide
```

We expect the output to be `healthy` for the first sample and `sick` for the second one.

Using the `predict` function we can retrieve the labels that the tree assigns to these samples.

```@example 1
prediction = predict(tree, test_data)
```