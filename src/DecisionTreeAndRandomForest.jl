module DecisionTreeAndRandomForest

using DocStringExtensions
using StatsBase: mode, sample, mean, countmap


include("DecisionTree.jl")
include("RandomForest.jl")
include("split-criterions/GiniImpurity.jl")
include("split-criterions/InformationGain.jl")
include("split-criterions/VarianceReduction.jl")
include("split-criterions/Utils.jl")

export DecisionTree, RandomForest, fit!, predict
export split_gini, split_ig, split_variance
export get_split_criterions
end
