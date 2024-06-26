using DecisionTreeAndRandomForest
using Test

include("../src/GiniImpurity.jl")
include("../src/InformationGain.jl")


@testset "DecisionTreeAndRandomForest.jl" begin
    include("test_informationgain.jl")
    include("test_classificationtree.jl")
    include("test_giniimpurity.jl")
    include("test_randomforest.jl")
    include("test_regressiontree.jl")
end


