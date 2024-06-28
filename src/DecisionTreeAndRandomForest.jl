module DecisionTreeAndRandomForest



# Write your package code here.
include("DecisionTree.jl")
include("RandomForest.jl")
include("split-criterions/GiniImpurity.jl")
include("split-criterions/InformationGain.jl")
include("split-criterions/VarianceReduction.jl")

export DecisionTree, fit, predict, print_tree, split_gini, split_ig, split_variance, RandomForest, print_forest
end
