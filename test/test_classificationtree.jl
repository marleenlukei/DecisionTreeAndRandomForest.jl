using DecisionTreeAndRandomForest
using Test
using MLJ: load_iris,unpack,partition
using DataFrames

@testset "ClassificationTree" begin
    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
    # data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    # labels = ["healthy", "healthy", "healthy", "healthy", "sick", "sick", "sick", "sick"]
    tree = ClassificationTree(data, labels)
    fit(tree)
    test_data = ["dog" 38.0; "human" 38.0]
    prediction = predict(tree, test_data)
    print_tree(tree)
    @test prediction[1] == "healthy"
    @test prediction[2] == "sick"

    data = load_iris()
	iris = DataFrame(data)
	y, X = unpack(iris, ==(:target); rng=123)
	train, test = partition(eachindex(y), 0.7)
	train_labels = Vector{String}(y[train])
	test_labels = Vector{String}(y[test])
	train_data = Matrix(X[train, :])
	test_data = Matrix(X[test, :])
    tree = ClassificationTree(train_data, train_labels)
    fit(tree)
    predictions = predict(tree, test_data)
    print_tree(tree)
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    println("Accuracy: $accuracy")
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90

end