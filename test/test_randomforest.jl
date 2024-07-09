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
    forest = RandomForest(5, 4, split_gini, 20, 0.8, 3)
    fit!(forest, train_data, train_labels)
    predictions = predict(forest, test_data)
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    
    @test Set(predictions) <= Set(test_labels)
    @test accuracy >= 0.90
    
    data = dataset("MASS", "biopsy")
    data = dropmissing(data)  
    X = data[:, 2:end-1]  
    y = data[:, end]
    train_indices, test_indices = partition(eachindex(y), 0.7, rng=123)
    train_labels = Vector{String}(y[train_indices])
    test_labels = Vector{String}(y[test_indices])
    train_data = Matrix(X[train_indices, :])
    test_data = Matrix(X[test_indices, :])
    forest = RandomForest(split_ig, 10)
    fit!(forest, train_data, train_labels)  
    predictions = predict(forest, test_data)  
    accuracy = sum(predictions .== test_labels) / length(test_labels)
    

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

    # Generate synthetic data
    n, m = 1000, 10
    features = randn(n, m)
    weights = rand(-2:2, m)
    labels = features * weights
    train_indices, test_indices = partition(eachindex(labels), 0.7, rng=123)
    train_data = features[train_indices, :]
    test_data = features[test_indices, :]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    forest = RandomForest(split_variance, 20, 0.8, 6)
    fit!(forest, train_data, train_labels)  
    predictions = predict(forest, test_data)
    ss_res = sum((test_labels .- predictions) .^ 2)
    ss_tot = sum((test_labels .- mean(test_labels)) .^ 2)
    r2_score = 1 - (ss_res / ss_tot)

    @test r2_score >= 0.70

end

@testset "RegressionForest" begin
    boston = dataset("MASS", "Boston")
    data = DataFrame(boston)
    X = data[:, 1:end-1]
    y = data[:, end]
    train_indices, test_indices = partition(eachindex(y), 0.95, rng=123)
    train_labels = Vector{Float64}(y[train_indices])
    test_labels = Vector{Float64}(y[test_indices])
    train_data = Matrix(X[train_indices, :])
    test_data = Matrix(X[test_indices, :])
    forest = RandomForest(split_ig, 10, 0.9, 10)
    fit!(forest, train_data, train_labels)  
    predictions = predict(forest, test_data)
    mse = mean((predictions .- test_labels) .^ 2)
    ss_res = sum((test_labels .- predictions) .^ 2)
    ss_tot = sum((test_labels .- mean(test_labels)) .^ 2)
    r2_score = 1 - (ss_res / ss_tot)

    @test mse <= 10.0
    @test r2_score >= 0.75

end