module DecisionTreeAndRandomForest



# Write your package code here.
include("ClassificationTree.jl")
include("GiniImpurity.jl")
include("Information_Gain_Arka.jl")
export ClassificationTree, fit, predict, print_tree, find_best_split
end
