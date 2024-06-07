using DecisionTreeAndRandomForest
using Test

@testset "DecisionTreeAndRandomForest.jl" begin
    include("Information_Gain_test.jl")
    include("test_classificationtree.jl")
    include("test_giniimpurity.jl")
end


