using DecisionTreeAndRandomForest
using Test
using MLJ: load_iris,unpack,partition
using DataFrames

@testset "RandomForest" begin
    data = load_iris()
	iris = DataFrame(data)
	y, X = unpack(iris, ==(:target); rng=123)
	train, test = partition(eachindex(y), 0.7)
	train_labels = Vector{String}(y[train])
	test_labels = Vector{String}(y[test])
	train_data = Matrix(X[train, :])
	test_data = Matrix(X[test, :])
    forest = RandomForest(train_data, train_labels, 5, 4, 20, 0.8)
    fit(forest, 3)
    predictions = predict(forest, test_data)
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    println("Accuracy: $accuracy")
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
end