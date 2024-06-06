using DecisionTreeAndRandomForest
using Test

@testset "entropy" begin
    labels = [1, 1, 0, 1, 0, 0]
    @test entropy(labels) ≈ 1.0

    labels = [1, 1, 1, 1, 1, 1]
    @test entropy(labels) ≈ 0.0

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    @test entropy(labels) ≈ -(8/10 * log2(8/10) + 2/10 * log2(2/10))

    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    @test entropy(labels) ≈ -(4/10 * log2(4/10) + 6/10 * log2(6/10))
end

@testset "information_gain" begin
    y = [1, 1, 0, 1, 0, 0]
    y_left = [1, 1, 0]
    y_right = [1, 0, 0]
    @test information_gain(y, y_left, y_right) ≈ 0.0

    y = [1, 1, 1, 0, 0, 0]
    y_left = [1, 1, 1]
    y_right = [0, 0, 0]
    @test information_gain(y, y_left, y_right) ≈ 1.0

    y = [1, 1, 1, 1, 0, 0, 0, 0]
    y_left = [1, 1, 1, 1]
    y_right = [0, 0, 0, 0]
    @test information_gain(y, y_left, y_right) ≈ 1.0

    y = [1, 1, 0, 0, 0, 0, 1, 1]
    y_left = [1, 1, 0, 0]
    y_right = [0, 0, 1, 1]
    @test information_gain(y, y_left, y_right) ≈ 0.0
end

@testset "find_best_split_ig" begin
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
    Y = ['N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y']  
    feature_index, feature_value = find_best_split(X, Y)  
    @test feature_index == 2  
    
    X = ["tech" "professional";
                "fashion" "student";
                "fashion" "professional";
                "sports" "student";
                "tech" "student";
                "tech" "retired";
                "sports" "professional"]
    Y = [1, 1, 1, 0, 0, 1, 0] 
    feature_index, feature_value = find_best_split(X, Y)
    @test feature_index == 1
    @test feature_value == "sports"

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

    Y = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
    feature_index, feature_value = find_best_split(X, Y) 
    @test feature_index == 1  
    @test feature_value == "middle_age"
end
