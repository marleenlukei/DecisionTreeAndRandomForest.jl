using DecisionTreeAndRandomForest
using Test

include("../src/GiniImpurity.jl")
include("../src/InformationGain.jl")


@testset "DecisionTreeAndRandomForest.jl" begin
    include("test_informationgain.jl")
    include("test_classificationtree.jl")
    include("test_giniimpurity.jl")
end


