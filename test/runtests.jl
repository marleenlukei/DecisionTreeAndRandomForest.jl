using DecisionTreeAndRandomForest
using Test
using MLJ: load_iris,unpack,partition
using DataFrames: DataFrame
using RDatasets: dataset 
using StatsBase: mean

@testset "DecisionTreeAndRandomForest.jl" begin
    include("test_informationgain.jl")
    include("test_giniimpurity.jl")
    include("test_classificationtree.jl")
    include("test_randomforest.jl")
    include("test_regressiontree.jl")
end


