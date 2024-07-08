# Test cases for entropy
@testset "Entropy" begin
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 1, 1, 1]) == 0
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 0, 1, 0]) ≈ 1
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 1, 0, 0, 0, 1, 1, 0]) ≈ 1
end

# Test cases for information_gain
@testset "Weighted Gain" begin
    y = [1, 1, 0, 0, 0, 1, 1, 0]
    y_left = [1, 1, 0, 0]
    y_right = [0, 1, 1, 0]
    @test DecisionTreeAndRandomForest.weighted_gain(y, y_left, y_right) ≈ 0.0

    y_left = [1, 1, 1, 1]
    y_right = [0, 0, 0, 0]
    @test DecisionTreeAndRandomForest.weighted_gain(y, y_left, y_right) ≈ 1.0
end


@testset "Information Gain" begin
 

    X = [1.0 2.0; 3.0 4.0; 2.5 0.5; 4.0 3.0]
    y = [0, 1, 0, 1]
    best_feature, best_threshold = DecisionTreeAndRandomForest.information_gain(X, y)
    @test best_feature == 1
    @test best_threshold == 3.0
end
