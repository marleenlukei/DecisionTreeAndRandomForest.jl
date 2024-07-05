# Test cases for entropy
@testset "entropy" begin
    @test DecisionTreeAndRandomForest.entropy([1, 1, 1, 1]) == 0
    @test DecisionTreeAndRandomForest.entropy([1, 0, 1, 0]) ≈ 1
    @test DecisionTreeAndRandomForest.entropy([1, 1, 0, 0, 0, 1, 1, 0]) ≈ 1
end

# Test cases for information_gain
@testset "information_gain" begin
    y = [1, 1, 0, 0, 0, 1, 1, 0]
    y_left = [1, 1, 0, 0]
    y_right = [0, 1, 1, 0]
    @test DecisionTreeAndRandomForest.information_gain(y, y_left, y_right) ≈ 0.0

    y_left = [1, 1, 1, 1]
    y_right = [0, 0, 0, 0]
    @test DecisionTreeAndRandomForest.information_gain(y, y_left, y_right) ≈ 1.0
end

# Test cases for split_dataset
@testset "split_dataset" begin
    X = [1.0 2.0; 3.0 4.0; 1.5 0.5; 3.5 3.5]
    y = [0, 1, 0, 1]
    X_left_expected = [1.0 2.0; 1.5 0.5]
    y_left_expected = [0, 0]
    X_right_expected = [3.0 4.0; 3.5 3.5]
    y_right_expected = [1, 1]

    X_left, y_left, X_right, y_right = DecisionTreeAndRandomForest.split_dataset(X, y, 1, 2.0)
    @test X_left == X_left_expected
    @test y_left == y_left_expected
    @test X_right == X_right_expected
    @test y_right == y_right_expected
end

# Test cases for best_split
@testset "best_split" begin
    X = [1.0 2.0; 3.0 4.0; 2.5 0.5; 4.0 3.0]
    y = [0, 1, 0, 1]
    best_feature, best_threshold = DecisionTreeAndRandomForest.best_split(X, y, 1)
    @test best_feature == 1
    @test best_threshold == 2.5
end
