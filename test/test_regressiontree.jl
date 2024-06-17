using DecisionTreeAndRandomForest
using Test
using MLJ:unpack,partition
using RDatasets:dataset 
using DataFrames  
using Statistics:mean

@testset "RegressionTree" begin
    boston = dataset("MASS", "Boston") 
    data = DataFrame(boston)
    X = data[:, 1:end-1]
    y = data[:, end]
    train_indices, test_indices = partition(eachindex(y), 0.95, rng=123)
    train_labels = Vector{Float64}(y[train_indices])
    test_labels = Vector{Float64}(y[test_indices])
    train_data = Matrix(X[train_indices, :])
    test_data = Matrix(X[test_indices, :])
    tree = ClassificationTree(train_data, train_labels,split_variance)
    fit(tree)
    predictions = predict(tree, test_data)
    mse = mean((predictions .- test_labels).^2)
    println("Mean Squared Error: ", mse)
    ss_res = sum((test_labels .- predictions).^2)
    ss_tot = sum((test_labels .- mean(test_labels)).^2)
    r2_score = 1 - (ss_res / ss_tot)
    println("R² Score: ", r2_score)
  
    @test mse <= 10.0  
    @test r2_score >= 0.75  
end