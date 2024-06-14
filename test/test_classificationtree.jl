using DecisionTreeAndRandomForest
using Test
using MLJ: load_iris, unpack, partition
using DataFrames

@testset "ClassificationTree" begin
    # Test 1: Categorical and numerical data with labels
    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
    
    # Using Information Gain criterion  
    split_criterion = split_gini
    tree = ClassificationTree(data, labels, split_criterion)
    fit(tree)
    
    test_data = ["dog" 38.0; "human" 38.0]
    prediction = predict(tree, test_data)
    print_tree(tree)
    @test prediction[1] == "healthy"
    @test prediction[2] == "sick"

    # Test 2: Using Iris dataset
    data = load_iris()
    iris = DataFrame(data)
    y, X = unpack(iris, ==(:target); rng=123)
    train, test = partition(eachindex(y), 0.7)
    train_labels = Vector{String}(y[train])
    test_labels = Vector{String}(y[test])
    train_data = Matrix(X[train, :])
    test_data = Matrix(X[test, :])

    tree = ClassificationTree(train_data, train_labels, split_criterion)
    fit(tree)
    predictions = predict(tree, test_data)
    print_tree(tree)
    
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
end
