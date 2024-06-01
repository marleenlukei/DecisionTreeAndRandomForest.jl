using DecisionTreeAndRandomForest
include("../src/GiniImpurity.jl")
using Test

@testset "DecisionTreeAndRandomForest.jl" begin
    # Write your tests here.
end


@testset "gini_impurity" begin
    labels = [1, 1, 0, 1, 0, 0]
    @test gini_impurity(labels) ≈ 0.5

    labels = [1, 1, 1, 1, 1, 1]
    @test gini_impurity(labels) ≈ 0.0

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  
    @test gini_impurity(labels) ≈ 1.0 - (8/10)^2 - (2/10)^2  
    
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  
    @test gini_impurity(labels) ≈ 1.0 - (4/10)^2 - (6/10)^2 

end

@testset "weighted_gini" begin
    left_dataset = [1, 1, 0, 0, 0]  
    right_dataset = [1, 1, 1, 0, 0]  
    @test weighted_gini(left_dataset, right_dataset) ≈ 0.48

    left_dataset = [0, 0, 0, 0, 0]
    right_dataset = [1, 1, 1, 1, 1]
    @test weighted_gini(left_dataset, right_dataset) ≈ 0.0
end


@testset "find_best_split" begin  
    Emotion=["sick","sick","notsick","notsick","notsick","sick","notsick","notsick"]
    Temperature = ["under","over","under","under","over","over","under","over"]
    StayHome=['N','Y','Y','N','Y','N','N','Y']
    Input = hcat(Emotion, Temperature)  
    StayHome = ['N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y']  
    feature_index1, feature_value1 = find_best_split(Input, StayHome)  
    @test feature_index1 == 2  
    @test feature_value1 == "over" 

    Feature1 = [1.0, 2.0, 3.0, 3.0, 2.0]  
    Feature2 = [2.0, 3.0, 4.0, 5.0, 6.0] 
    Input = hcat(Feature1, Feature2)   
    Label = [0, 0, 1, 1, 0]  
    feature_index2, feature_value2 = find_best_split(Input, Label)  
    @test feature_index2 == 1  
    @test feature_value2 == 2.0   

end