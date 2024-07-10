@testset "ClassificationTree" begin
    # Test 1: Categorical and numerical data with labels
    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]

    # Using Information Gain criterion  

    tree = DecisionTree(split_gini)
    fit!(tree, data, labels)

    test_data = ["dog" 38.0; "human" 38.0]
    prediction = predict(tree, test_data)
    print(tree)
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

    tree = DecisionTree(split_gini)
    fit!(tree, train_data, train_labels)
    predictions = predict(tree, test_data)
    print(tree)

    accuracy = sum(predictions .== test_labels) / length(test_labels)
    println("Accuracy: $accuracy")
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
end

@testset "print DecisionTree" begin
    tree = DecisionTree(split_gini)

    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]

    fit!(tree, data, labels)

    result = @capture_out print(tree)
    expected = """Feature: 2, Split Value: 37.4
   ├── Labels: healthy (3/3) 
   └── Feature: 1, Split Value: dog
       ├── Labels: sick (2/2) 
       └── Feature: 2, Split Value: 40.2
           ├── Labels: healthy (2/2) 
           └── Labels: sick (1/1) """
    @test chomp(result) == chomp(expected)

    emptytree = DecisionTree(split_gini)
    result = @capture_out print(emptytree)
    @test chomp(result) == "missing"
end
