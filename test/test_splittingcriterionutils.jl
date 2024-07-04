@testset "get_split_criterions" begin
    @test get_split_criterions() == (split_gini, split_ig, split_variance)
    @test get_split_criterions("42") == (split_gini, split_ig, split_variance)
    @test get_split_criterions("regression") == (split_variance,)
    @test get_split_criterions("classification") == (split_gini, split_ig)
end