
# Decision Trees and Random Forests on Real-World Datasets


## Classification Tree and Random Forest - Classifying Iris Types

In this tutorial, we will demonstrate how to use the package to create a classification tree and a random forest, and apply them to classify the Iris dataset.

First, ensure to import all the necessary packages:

```julia
using DataFrames
using MLJ: load_iris, unpack, partition
using DecisionTreeAndRandomForest
```

Second, we load the Iris dataset and prepare the data by splitting it into training and test sets:

```julia
data = load_iris()
iris = DataFrame(data)
y, X = unpack(iris, ==(:target); rng=123)
train, test = partition(eachindex(y), 0.7)
train_labels = Vector{String}(y[train])
test_labels = Vector{String}(y[test])
train_data = Matrix(X[train, :])
test_data = Matrix(X[test, :])
```

### Decision Tree

Next, we create the decision tree and fit it to the training data:

```julia
# Initialize the classification tree with hyperparameters:
# - max_depth: Maximum depth of the tree (-1 means no limit).
# - min_samples_split: Minimum number of samples required to split an internal node.
# - num_features: Number of features to consider when looking for the best split (-1 means all features).
# - split_criterion: Function to measure the quality of a split.
tree = DecisionTree(-1, 1, -1, split_gini)
fit!(tree, train_data, train_labels)
```

Now we can use the Decision Tree to make predictions for unseen data:

```julia
predictions = predict(tree, test_data)
println("Decision Tree - Correct label: ", test_labels[1])
println("Decision Tree - Predicted label: ", predictions[1])
```

We can also test the accuracy of our Classification Tree:

```julia
accuracy = sum(predictions .== test_labels) / length(test_labels)
println("Decision Tree - Accuracy: ", accuracy)
```

### Random Forest

Next, we create the random forest and fit it to the training data:

```julia
# Initialize the random forest with hyperparameters:
# - max_depth: Maximum depth of the trees (-1 means no limit).
# - min_samples_split: Minimum number of samples required to split an internal node.
# - split_criterion: Function to measure the quality of a split.
# - number_of_trees: Number of trees in the forest.
# - subsample_percentage: Percentage of samples to use for each tree (0.8 means 80% of the data).
# - num_features: Number of features to consider when looking for the best split (-1 means all features).
forest = RandomForest(7, 3, split_gini, 15, 0.7, -1)  # Using 7 for max_depth, 3 for min_samples_split, 15 for number_of_trees, and 0.7 for subsample_percentage
fit!(forest, train_data, train_labels)
```

Now we can use the Random Forest to make predictions for unseen data:

```julia
forest_predictions = predict(forest, test_data)
println("Random Forest - Correct label: ", test_labels[1])
println("Random Forest - Predicted label: ", forest_predictions[1])
```

We can also test the accuracy of our Random Forest:

```julia
forest_accuracy = sum(forest_predictions .== test_labels) / length(test_labels)
println("Random Forest - Accuracy: ", forest_accuracy)
```

## Regression Tree and Random Forest - Predicting Housing Prices

We will now demonstrate how to use the package to create a regression tree and a random forest, and apply them to the Boston Housing dataset. The dataset contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. It is often used to predict the median value of owner-occupied homes (in $1000s).

First, ensure to import all the necessary packages:

```julia
using MLJ: unpack, partition
using RDatasets: dataset
using DataFrames
using DecisionTreeAndRandomForest
using Statistics: mean
```

Second, we load the Boston Housing dataset and prepare the data by splitting it into training and test sets:

```julia
boston = dataset("MASS", "Boston")
data = DataFrame(boston)
X = data[:, 1:end-1]
y = data[:, end]
train_indices, test_indices = partition(eachindex(y), 0.95, rng=123)
train_labels = Vector{Float64}(y[train_indices])
test_labels = Vector{Float64}(y[test_indices])
train_data = Matrix(X[train_indices, :])
test_data = Matrix(X[test_indices, :])
```

### Decision Tree

Next, we create the decsion tree and fit it to the training data:

```julia
tree = DecisionTree(-1, 1, -1, split_variance)
fit!(tree, train_data, train_labels)
```

Now we can use the Regression Tree to make predictions for unseen data:

```julia
predictions = predict(tree, test_data)
```

We can also assess the quality of our Regression Tree:

```julia
mse = mean((predictions .- test_labels).^2)
println("Decision Tree - Mean Squared Error: ", mse)
ss_res = sum((test_labels .- predictions).^2)
ss_tot = sum((test_labels .- mean(test_labels)).^2)
r2_score = 1 - (ss_res / ss_tot)
println("Decision Tree - R² Score: ", r2_score)
```

### Random Forest

Next, we create the random forest and fit it to the training data:

```julia
forest = RandomForest(-1, 1, split_variance, 10, 0.8, -1)
fit!(forest, train_data, train_labels)
```

Now we can use the Random Forest to make predictions for unseen data:

```julia
forest_predictions = predict(forest, test_data)
```

We can also assess the quality of our Random Forest:

```julia
forest_mse = mean((forest_predictions .- test_labels).^2)
println("Random Forest - Mean Squared Error: ", forest_mse)
forest_ss_res = sum((test_labels .- forest_predictions).^2)
forest_ss_tot = sum((test_labels .- mean(test_labels)).^2)
forest_r2_score = 1 - (forest_ss_res / forest_ss_tot)
println("Random Forest - R² Score: ", forest_r2_score)
```

