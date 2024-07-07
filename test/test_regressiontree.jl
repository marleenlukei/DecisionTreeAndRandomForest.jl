@testset "RegressionTree" begin
    data = DataFrame(
        FloorArea=[14, 20, 25, 33, 40, 55, 80],
        Rooms=[1, 1, 1, 2, 2, 3, 4],
        YearBuilt=[1990, 1980, 2000, 2005, 2010, 1999, 2020],
        Amenities=["Standard", "Modern", "Luxury", "Basic", "Modern", "Standard", "Luxury"],
        Rent=[500, 800, 1500, 1100, 2200, 2000, 3000]
    )
    X = Matrix(data[:, 1:2])
    y = data[:, :Rent]

    tree = DecisionTree(3, 3, split_variance)
    fit!(tree, X, y)

    test_data = [12 1 2002 "Standard"; 40 2 2020 "Luxury"]
    prediction = predict(tree, test_data)
    print(tree)
    print(prediction)
    @test 600 <= prediction[1] <= 700
    @test 2000 <= prediction[2] <= 2500


    boston = dataset("MASS", "Boston")
    data = DataFrame(boston)
    X = data[:, 1:end-1]
    y = data[:, end]
    train_indices, test_indices = partition(eachindex(y), 0.95, rng=123)
    train_labels = Vector{Float64}(y[train_indices])
    test_labels = Vector{Float64}(y[test_indices])
    train_data = Matrix(X[train_indices, :])
    test_data = Matrix(X[test_indices, :])
    
    tree = DecisionTree(split_variance)
    fit!(tree, train_data, train_labels)
    
    predictions = predict(tree, test_data)

    ss_res = sum((test_labels .- predictions) .^ 2)
    ss_tot = sum((test_labels .- mean(test_labels)) .^ 2)
    r2_score = 1 - (ss_res / ss_tot)
    println("R² Score: ", r2_score)

    @test r2_score >= 0.70
end