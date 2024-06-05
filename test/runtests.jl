using Random
using Statistics
using StatsBase  # Importing StatsBase for the countmap function
include("../Information_Gain_Arka.jl")  # Assuming your functions are in information_gain.jl

# Test Functions

function test_entropy()
    y = [1, 1, 1, 0, 0, 0]
    calculated_entropy = entropy(y)
    expected_entropy = 1.0
    if abs(calculated_entropy - expected_entropy) < 1e-5
        println("test_entropy: Passed")
    else
        println("test_entropy: Failed")
    end
end

function test_information_gain()
    y = [1, 1, 1, 0, 0, 0]
    y_left = [1, 1, 1]
    y_right = [0, 0, 0]
    calculated_gain = information_gain(y, y_left, y_right)
    expected_gain = 1.0
    if abs(calculated_gain - expected_gain) < 1e-5
        println("test_information_gain: Passed")
    else
        println("test_information_gain: Failed")
    end
end

function test_split_dataset()
    X = [2.0 3.0; 3.0 4.0; 6.0 7.0; 8.0 9.0; 1.0 2.0; 7.0 8.0]
    y = [0, 0, 1, 1, 0, 1]
    feature = 1
    threshold = 5.0
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
    if all(X_left[:, feature] .<= threshold) && all(X_right[:, feature] .> threshold)
        println("test_split_dataset: Passed")
    else
        println("test_split_dataset: Failed")
    end
end

function test_best_split()
    X = [2.0 3.0; 3.0 4.0; 6.0 7.0; 8.0 9.0; 1.0 2.0; 7.0 8.0]
    y = [0, 0, 1, 1, 0, 1]
    feature, threshold = best_split(X, y)
    if feature > 0 && threshold > 0
        println("test_best_split: Passed")
    else
        println("test_best_split: Failed")
    end
end

# Run Tests
println("Running Tests for Information Gain Functions...")
test_entropy()
test_information_gain()
test_split_dataset()
test_best_split()
println("All tests completed.")