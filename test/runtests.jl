using DecisionTreeAndRandomForest
using Test

include("../src/split-criterions/GiniImpurity.jl")
include("../src/split-criterions/InformationGain.jl")
include("../src/split-criterions/VarianceReduction.jl")


@testset "DecisionTreeAndRandomForest.jl" begin
    include("test_informationgain.jl")
    include("test_classificationtree.jl")
    include("test_giniimpurity.jl")
    include("test_randomforest.jl")
    include("test_regressiontree.jl")
end


