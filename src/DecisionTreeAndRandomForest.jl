module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("RandomForest.jl")
include("GiniImpurity.jl")
include("InformationGain.jl")

export ClassificationTree, fit, predict, print_tree, find_best_split, RandomForest, print_forest
end
