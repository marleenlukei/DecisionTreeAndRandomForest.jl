# Decision Trees on Real-World Datasets


## Classification Tree - Classifying iris types
In this tutorial, we will demonstrate how to use the package to create a classification tree and apply it to classify the Iris dataset.  

First, ensure to import all the necessary packages: 

```@example 2
using DataFrames  
using MLJ: load_iris,unpack,partition
using DecisionTreeAndRandomForest
```
Second, we load the Iris dataset and prepare the data by splitting into training and test sets:

```@example 2
data = load_iris()
iris = DataFrame(data)  
y, X = unpack(iris, ==(:target); rng=123)   
train, test = partition(eachindex(y), 0.7)  
train_labels = Vector{String}(y[train])  
test_labels = Vector{String}(y[test])  
train_data = Matrix(X[train, :])  
test_data = Matrix(X[test, :])  
nothing # hide
```

Next, we create the classification tree and fit it to the training data:

```@example 2
tree = ClassificationTree(train_data, train_labels, split_gini)
fit(tree)  
nothing # hide
```

Now we can use the Classification Tree to make predictions for unseen data:

```@example 2
predictions = predict(tree, test_data)  
println("Correct label: ", test_labels[1])  
println("Predicted label: ", predictions[1])  
```

We can also test the accuracy of our Classification Tree:

```@example 2
accuracy = sum(predictions .== test_labels) / length(test_labels)  
println("Accuracy: ", accuracy)  
```





## Regression Tree - Predicting housing prices 

In this tutorial, we will demonstrate how to use the package to create a regression tree and apply it to the Boston Housing dataset.  
The dataset contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. It is often used to predict the median value of owner-occupied homes (in \$1000s).

First, ensure to import all the necessary packages: 

```@example 3
using MLJ:unpack,partition
using RDatasets:dataset 
using DataFrames
using DecisionTreeAndRandomForest
using Statistics:mean
```
Second, we load the Boston Housing dataset and prepare the data by splitting into training and test sets:

```@example 3
boston = dataset("MASS", "Boston") 
data = DataFrame(boston)
X = data[:, 1:end-1]
y = data[:, end]
train_indices, test_indices = partition(eachindex(y), 0.95, rng=123)
train_labels = Vector{Float64}(y[train_indices])
test_labels = Vector{Float64}(y[test_indices])
train_data = Matrix(X[train_indices, :])
test_data = Matrix(X[test_indices, :]) 
nothing # hide
```

Next, we create the Regression Tree and fit it to the training data:

```@example 3
tree = ClassificationTree(train_data, train_labels, split_variance)
fit(tree)  
nothing # hide
```

Now we can use the Regression Tree to make predictions for unseen data:

```@example 3
predictions = predict(tree, test_data) 
nothing # hide
```

We can also assess the quality of our Regression Tree:

```@example 3
mse = mean((predictions .- test_labels).^2)
println("Mean Squared Error: ", mse)
ss_res = sum((test_labels .- predictions).^2)
ss_tot = sum((test_labels .- mean(test_labels)).^2)
r2_score = 1 - (ss_res / ss_tot)
println("R² Score: ", r2_score)
```



