# Test cases for entropy
@testset "calculate_entropy" begin
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 1, 1, 1]) == 0
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 0, 1, 0]) ≈ 1
    @test DecisionTreeAndRandomForest.calculate_entropy([1, 1, 0, 0, 0, 1, 1, 0]) ≈ 1
end

# Test cases for weighted_entropy
@testset "weighted_entropy" begin
    y_left = [1, 1, 0, 0]
    y_right = [0, 1, 1, 0]
    @test DecisionTreeAndRandomForest.weighted_entropy(y_left, y_right) ≈ 1.0

    y_left = [1, 1, 1, 1]
    y_right = [0, 0, 0, 0]
    @test DecisionTreeAndRandomForest.weighted_entropy(y_left, y_right) ≈ 0.0
end


@testset "information_gain" begin
 

    X = [1.0 2.0; 3.0 4.0; 2.5 0.5; 4.0 3.0]
    y = [0, 1, 0, 1]
    best_feature, best_threshold = DecisionTreeAndRandomForest.information_gain(X, y)
    @test best_feature == 1
    @test best_threshold == 3.0
end

@testset "split_ig_categorical" begin
    X = [
        "low" "blue";
        "low" "blue";
        "medium" "red";
        "high" "blue"
    ]
    y = [1, 1, 2, 2]

    feature_index, feature_value = DecisionTreeAndRandomForest.split_ig(X, y)

    @test feature_index == 1 
    @test feature_value == "low"  
end
