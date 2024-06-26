module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("RandomForest.jl")
include("GiniImpurity.jl")
include("InformationGain.jl")
include("VarianceReduction.jl")

export ClassificationTree, fit, predict, print_tree, split_gini, split_ig, split_variance, RandomForest, print_forest
end
