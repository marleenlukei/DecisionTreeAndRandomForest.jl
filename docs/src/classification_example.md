# Example Tutorial: Classification Tree

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
tree = ClassificationTree(train_data, train_labels)
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
