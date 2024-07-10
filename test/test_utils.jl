@testset "get_split_criterions" begin
    @test get_split_criterions() == (split_gini, split_ig, split_variance)
    @test get_split_criterions("42") == (split_gini, split_ig, split_variance)
    @test get_split_criterions("regression") == (split_variance,)
    @test get_split_criterions("classification") == (split_gini, split_ig)
end

@testset "split_node tests" begin
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
        left, right =  DecisionTreeAndRandomForest.split_node(data, labels, feature_index, split_value)
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
        left, right = DecisionTreeAndRandomForest.split_node(data_categorical, labels_categorical, feature_index, split_value)
        @test left == left_expected
        @test right == right_expected
    end
end