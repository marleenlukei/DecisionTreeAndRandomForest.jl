module DecisionTreeAndRandomForest

include("ClassificationTree.jl")
include("GiniImpurity.jl")

export ClassificationTree, fit, predict, print_tree, find_best_split

end
