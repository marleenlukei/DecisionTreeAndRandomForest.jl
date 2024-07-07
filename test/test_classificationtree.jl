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
    
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
end

@testset "fit! - exception handling" begin
    # Define mismatched data and labels
    data = rand(10, 3) 
    labels = rand(9)
    tree = DecisionTree(split_gini)
    @test_throws ArgumentError begin
        fit!(tree, data, labels)
    end

    try
        fit!(tree, data, labels)
    catch e
        @test occursin("The number of rows in data must match the number of elements in labels", string(e))
    end
end

@testset "predict - exception handling" begin
    test_data = Matrix{Float64}(undef, 0, 0)
    tree = DecisionTree(split_ig) 
    @test_throws MethodError begin
        predict(tree, test_data)
    end

    try
        predict(tree, test_data)
    catch e
        @test occursin("The tree needs to be fitted first!", string(e))
    end
end

@testset "build_tree - Edge Cases" begin
    @testset "Threshold not found" begin
        test_data = [1 2; 1  2; 1 2]
        labels = ["1", "2", "1"]
        tree = DecisionTree(split_ig) 
        fit!(tree, test_data, labels)
        test_data = [1 2]
        prediction = predict(tree, test_data)
        print(tree)
    
        @test prediction[1] == "1"
    end

    @testset "Empty Subtree" begin
        data_empty_subtree = [1.0 2.0 3.0; 
                            4.0 5.0 6.0]
        labels_empty_subtree = [1.0, 2.0]
        
        function dummy_split_criterion(data::AbstractMatrix, labels::AbstractVector, num_features::Int=-1)
            return 1, 1.0
        end
    
        tree = DecisionTree(-1, 1, -1, dummy_split_criterion)
        fit!(tree, data_empty_subtree, labels_empty_subtree)
        print(tree)
    
        # Check if the root is a Leaf due to empty subtree
        @test isa(tree.root, DecisionTreeAndRandomForest.Leaf)
        @test tree.root.values == labels_empty_subtree
    end
end
