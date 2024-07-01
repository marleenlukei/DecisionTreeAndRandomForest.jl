@testset "RandomForest" begin
    data = load_iris()
    iris = DataFrame(data)
    y, X = unpack(iris, ==(:target); rng=123)
    train, test = partition(eachindex(y), 0.7)
    train_labels = Vector{String}(y[train])
    test_labels = Vector{String}(y[test])
    train_data = Matrix(X[train, :])
    test_data = Matrix(X[test, :])
    forest = RandomForest(5, 4, split_gini, 20, 0.8, 3)
    fit!(forest, train_data, train_labels)
    print(forest)
    predictions = predict(forest, test_data)
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    println("Accuracy: $accuracy")
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
end

@testset "RandomRegressionForest" begin
    data = DataFrame(
        FloorArea=[14, 20, 25, 33, 40, 55, 80],
        Rooms=[1, 1, 1, 2, 2, 3, 4],
        YearBuilt=[1990, 1980, 2000, 2005, 2010, 1999, 2020],
        Amenities=["Standard", "Modern", "Luxury", "Basic", "Modern", "Standard", "Luxury"],
        Rent=[500, 800, 1500, 1100, 2200, 2000, 3000]
    )
    X = Matrix(data[:, 1:2])
    y = data[:, :Rent]
    forest = RandomForest(split_variance)
    fit!(forest, X, y)
    test_data = [12 1 2002 "Standard"; 40 2 2020 "Luxury"]
    predictions = predict(forest, test_data)

    @test 500 <= predictions[1] <= 1200
    @test 1500 <= predictions[2] <= 2500
end