using DecisionTreeAndRandomForest
using Test

@testset "DecisionTreeAndRandomForest.jl" begin
    include("test_classificationtree.jl")
    include("test_giniimpurity.jl")
end


