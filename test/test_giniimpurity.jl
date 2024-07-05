@testset "gini_impurity" begin
    labels = [1, 1, 0, 1, 0, 0]
    @test DecisionTreeAndRandomForest.gini_impurity(labels) ≈ 0.5

    labels = [1, 1, 1, 1, 1, 1]
    @test DecisionTreeAndRandomForest.gini_impurity(labels) ≈ 0.0

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  
    @test DecisionTreeAndRandomForest.gini_impurity(labels) ≈ 1.0 - (8/10)^2 - (2/10)^2  
    
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  
    @test DecisionTreeAndRandomForest.gini_impurity(labels) ≈ 1.0 - (4/10)^2 - (6/10)^2 

end

@testset "weighted_gini" begin
    left_dataset = [1, 1, 0, 0, 0]  
    right_dataset = [1, 1, 1, 0, 0]  
    @test DecisionTreeAndRandomForest.weighted_gini(left_dataset, right_dataset) ≈ 0.48

    left_dataset = [0, 0, 0, 0, 0]
    right_dataset = [1, 1, 1, 1, 1]
    @test DecisionTreeAndRandomForest.weighted_gini(left_dataset, right_dataset) ≈ 0.0
end


@testset "find_best_split" begin  
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
    feature_index, feature_value = DecisionTreeAndRandomForest.find_best_split(X, Y)  
    @test feature_index == 2  
    
    X = ["tech" "professional";
                "fashion" "student";
                "fashion" "professional";
                "sports" "student";
                "tech" "student";
                "tech" "retired";
                "sports" "professional"]
    Y = [1, 1, 1, 0, 0, 1, 0] 
    feature_index, feature_value = DecisionTreeAndRandomForest.find_best_split(X, Y)
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
    feature_index, feature_value = DecisionTreeAndRandomForest.find_best_split(X, Y) 
    @test feature_index == 1  
    @test feature_value == "middle_age"
 


end

@testset "split_gini_numerical" begin
    data = [
        1 2;
        1 2;
        2 2;
        2 3;
        3 3;
        3 3
    ]
    labels = ["A", "A", "B", "B", "B", "A"]

    feature_index, feature_value = split_gini(data, labels)

    @test feature_index == 1
    @test feature_value == 2
end

@testset "split_gini_categorical" begin
    data = [
        "low" "blue";
        "low" "blue";
        "medium" "red";
        "high" "blue"
    ]
    labels = [1, 1, 2, 2]

    feature_index, feature_value = split_gini(data, labels)

    @test feature_index == 1  
    @test feature_value == "low"  
end