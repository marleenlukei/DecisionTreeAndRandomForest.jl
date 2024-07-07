# Test the variance calculation
@testset "variance" begin
    data = [1.0, 2.0, 3.0, 4.0, 5.0]  
    @test DecisionTreeAndRandomForest.variance(data) == 2.5

    data = [42.0]
    @test isnan(DecisionTreeAndRandomForest.variance(data))

    data = [2.0, 2.0, 2.0, 2.0]
    @test DecisionTreeAndRandomForest.variance(data) == 0.0
    
end

# Test variance_reduction
@testset "variance_reduction tests" begin
    # Basic functionality
    left = [1.0, 2.0]
    right = [3.0, 4.0]
    @test DecisionTreeAndRandomForest.variance_reduction(left, right) == 0.5

    # Edge cases
    left_empty = Float64[]
    right = [1.0, 2.0, 3.0, 4.0]
    @test DecisionTreeAndRandomForest.variance_reduction(left_empty, right) ≈ 1.66 atol=1e-2
    
    left = [1.0, 2.0, 3.0, 4.0]
    right_empty = Float64[]
    @test DecisionTreeAndRandomForest.variance_reduction(left, right_empty) ≈ 1.66 atol=1e-2

end

@testset "split_node_vr tests" begin
    @testset "Numerical split" begin
        data = [1.0 2.0 3.0; 
                4.0 5.0 6.0;
                7.0 8.0 9.0;
                10.0 11.0 12.0]
        labels = [1.0, 2.0, 3.0, 4.0]
        feature_index = 2
        split_value = 5.0
        left_expected = [1.0]
        right_expected = [2.0, 3.0, 4.0]
        left, right =  DecisionTreeAndRandomForest.split_node_vr(data, labels, feature_index, split_value)
        @test left == left_expected
        @test right == right_expected
    end

    @testset "Categorical split" begin
        data_categorical = ["red" "circle"; 
                            "blue" "square"; 
                            "red" "square"; 
                            "green" "circle"]
        labels_categorical = [1, 2, 3, 4]
        feature_index = 1
        split_value = "red"
        left_expected = [2, 4]
        right_expected = [1, 3]
        left, right = DecisionTreeAndRandomForest.split_node_vr(data_categorical, labels_categorical, feature_index, split_value)
        @test left == left_expected
        @test right == right_expected
    end
end

# Test find_best_split_vr
@testset "find_best_split_vr tests" begin
    data = [1.0 2.0 3.0; 
            4.0 5.0 6.0;
            7.0 8.0 9.0;
            10.0 11.0 12.0]
    labels = [1.0, 2.0, 3.0, 4.0]

    @testset "Basic functionality" begin
        best_feature, best_threshold = DecisionTreeAndRandomForest.find_best_split_vr(data, labels)
        @test best_feature == 1 
        @test best_threshold == 7.0  
    end

    @testset "Edge cases" begin
        data_single_row = [1.0 2.0 3.0]
        labels_single_row = [1.0]
        best_feature, best_threshold = DecisionTreeAndRandomForest.find_best_split_vr(data_single_row, labels_single_row)
        @test best_feature == -1  
        @test best_threshold == -1
    end

    @testset "Categorical data" begin
        X = [
            "sick" "under";
            "sick" "over";
            "notsick" "under";
            "notsick" "under";
            "notsick" "over";
            "sick" "over";
            "notsick" "under";
            "notsick" "over"
        ]
        Y = [0, 1, 1, 0, 1, 0, 0, 1]
        feature_index, feature_value = DecisionTreeAndRandomForest.find_best_split_vr(X, Y) 
        @test feature_index == 2 
        @test feature_value == "under"
    end
end

# Test split_variance (wrapper function)
@testset "split_variance tests" begin

    X = [
        "youth" "high" "no" "fair";
        "youth" "high" "no" "excellent";
        "middle_age" "high" "no" "fair";
        "senior" "medium" "no" "fair";
        "senior" "low" "yes" "fair";
        "senior" "low" "yes" "excellent";
        "middle_age" "low" "yes" "excellent";
        "youth" "medium" "no" "fair";
        "youth" "low" "yes" "fair";
        "senior" "medium" "yes" "fair";
        "youth" "medium" "yes" "excellent";
        "middle_age" "medium" "no" "excellent";
        "middle_age" "high" "yes" "fair";
        "senior" "medium" "no" "excellent"
        ]

    Y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

    feature_index, feature_value = DecisionTreeAndRandomForest.find_best_split_vr(X, Y) 
    feature_index_split_variance, feature_value_split_variance = DecisionTreeAndRandomForest.split_variance(X, Y) 
    @test feature_index == feature_index_split_variance  
    @test feature_value == feature_value_split_variance
    
end
