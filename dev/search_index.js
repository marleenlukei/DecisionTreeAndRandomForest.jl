var documenterSearchIndex = {"docs":
[{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"This is a basic example on how to use the classification tree.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"First, import the module like this","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using DecisionTreeAndRandomForest","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Second we need some training data and their respective labels.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"data = [\"dog\" 37.0; \"dog\" 38.4; \"dog\" 40.2; \"dog\" 38.9; \"human\" 36.2; \"human\" 37.4; \"human\" 38.8; \"human\" 36.2]\nlabels = [\"healthy\", \"healthy\", \"sick\", \"healthy\", \"healthy\", \"sick\", \"sick\", \"healthy\"]\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"After that we can initialiate a tree. There are two constructors:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"One only takes the data and labels as parameters.\nThe other one can also take values for max_depth and min_samples_split.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"tree = ClassificationTree(data, labels)\nother_tree = ClassificationTree(3, 2, data, labels)\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We can build the tree using the fit function.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"info: Info\nAt the moment the fit function uses the Gini-Impurity to find the optimal split. In the future you can provide a custom function by passing it into the fit function.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"fit(tree)\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"To take a look at the tree, we can do the following:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"warning: Warning\nThis function is mainly used for debugging purposes. It could be removed in future releases.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"print_tree(tree)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Lastly, we want to classify some test samples. Therefore we need to create some.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"test_data = [\"dog\" 38.0; \"human\" 38.0]\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We expect the output to be healthy for the first sample and sick for the second one.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Using the predict function we can retrieve the labels that the tree assigns to these samples.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"prediction = predict(tree, test_data)","category":"page"},{"location":"splitting_criterion/#Splitting-Criterion","page":"Splitting Criterion","title":"Splitting Criterion","text":"","category":"section"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"This package offers multiple options of splitting criterion for evaluating the quality of a split and therefore constructing a decision tree. In the following a brief overview of the available criterion and their use cases is provided.","category":"page"},{"location":"splitting_criterion/#Gini-Impurity","page":"Splitting Criterion","title":"Gini Impurity","text":"","category":"section"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Gini Impurity measures the likelihood of an incorrect classification of a randomly chosen element if it is labeled according to the distribution of labels in the dataset.  Gini Impurity is primarily used in classification problems. A lower Gini impurity indicates a purer node with a higher confidence in predicting the class label. The goal is to minimize the Gini Impurity at each split, thereby creating nodes that are as pure as possible.","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Gini(S) = 1 - sum_i=1^C (p_i)^2","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"where:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"S is the set of data points in the current node.\n\nC is the number of classes.\n\np_i is the proportion of data points belonging to class i in S.\n","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Steps for Calculation:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Calculate Gini coefficients for each child node.\nCompute the impurity for each split using a weighted Gini score.\nChoose the split with the lowest Gini impurity.","category":"page"},{"location":"splitting_criterion/#Information-Gain","page":"Splitting Criterion","title":"Information Gain","text":"","category":"section"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Information Gain calculated as the difference in entropy before and after splitting a dataset on an attribute. Entropy measures the uncertainty or impurity in the data. The goal is to reduce entropy and maximize information gain, leading to a more informative split. Information Gain is used in classification problems to choose the attribute that provides the highest information gain, resulting in the most informative split.","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"The information gain for a dataset 𝑆 after a split on attribute 𝐴 is given by:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Gain(S A) = Entropy(S) - sum_v in textValues(A) fracS_vS Entropy(S_v)\n","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"where:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"𝐴 is a feature of the dataset.\n\nv is a specific value of the feature 𝐴.\n  \n","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Entropy is calculated as:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Entropy(S) = -sum_i=1^C p_i log_2(p_i)","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"where:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"S is the set of data points in the current node.\n\nC is the number of classes.\n\np_i is the proportion of data points belonging to class i in S.\n","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"\nSteps for Calculation:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Calculate the entropy of the original dataset 𝑆.\nFor each split on attribute 𝐴 calculate the entropy of each child node 𝑆_𝑣 and calculate the weighted entropy after the split.\nCompute the Information Gain by subtracting the weighted entropy from the original entropy.\nChoose the split with the highest information gain.","category":"page"},{"location":"splitting_criterion/#Variance-Reduction","page":"Splitting Criterion","title":"Variance Reduction","text":"","category":"section"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Variance Reduction measures the reduction in variance of the target variable achieved by splitting a node. Higher variance reduction indicates a more informative split. Variance reduction is particularly useful for regression problems where the goal is to predict a continuous target variable.","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"textVR(S) = sigma^2(S) - sum_i=1^n fracS_iS sigma^2(S_i)","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"where:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"sigma^2(S): Variance of the parent node S.\nS: Set of data points in the current node.\nS_i: Subsets of S after the split.\nn: Number of subsets after the split.","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Steps for Calculation:","category":"page"},{"location":"splitting_criterion/","page":"Splitting Criterion","title":"Splitting Criterion","text":"Calculate the variance of the parent node S.\nFor each child node S_i, calculate its variance.\nCompute the weighted sum of the variances of the child nodes S_i.\nSubtract the weighted sum from the variance of the parent node S to get the variance reduction.\nChoose the split with the highest variance reduction.","category":"page"},{"location":"classification_example/#Example-Tutorial:-Classification-Tree","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"","category":"section"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"In this tutorial, we will demonstrate how to use the package to create a classification tree and apply it to classify the Iris dataset.  ","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"First, ensure to import all the necessary packages: ","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"using DataFrames  \nusing MLJ: load_iris,unpack,partition\nusing DecisionTreeAndRandomForest","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"Second, we load the Iris dataset and prepare the data by splitting into training and test sets:","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"data = load_iris()\niris = DataFrame(data)  \ny, X = unpack(iris, ==(:target); rng=123)   \ntrain, test = partition(eachindex(y), 0.7)  \ntrain_labels = Vector{String}(y[train])  \ntest_labels = Vector{String}(y[test])  \ntrain_data = Matrix(X[train, :])  \ntest_data = Matrix(X[test, :])  \nnothing # hide","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"Next, we create the classification tree and fit it to the training data:","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"tree = ClassificationTree(train_data, train_labels)\nfit(tree)  \nnothing # hide","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"Now we can use the Classification Tree to make predictions for unseen data:","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"predictions = predict(tree, test_data)  \nprintln(\"Correct label: \", test_labels[1])  \nprintln(\"Predicted label: \", predictions[1])  ","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"We can also test the accuracy of our Classification Tree:","category":"page"},{"location":"classification_example/","page":"Example Tutorial: Classification Tree","title":"Example Tutorial: Classification Tree","text":"accuracy = sum(predictions .== test_labels) / length(test_labels)  \nprintln(\"Accuracy: \", accuracy)  ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = DecisionTreeAndRandomForest","category":"page"},{"location":"#DecisionTreeAndRandomForest","page":"Home","title":"DecisionTreeAndRandomForest","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DecisionTreeAndRandomForest.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DecisionTreeAndRandomForest]","category":"page"},{"location":"#DecisionTreeAndRandomForest.ClassificationTree","page":"Home","title":"DecisionTreeAndRandomForest.ClassificationTree","text":"Represents a ClassificationTree.\n\nmax_depth controls the maximum depth of the tree. If -1, the depth is not limited.\n\nmin_samples_split controls when a node in the decision tree should be split.\n\nroot contains the root Node of the ClassificationTree.\n\ndata contains the datapoints of the ClassificationTree.\n\nlabels contains the respective labels of the datapoints.\n\n\n\n\n\n","category":"type"},{"location":"#DecisionTreeAndRandomForest.Leaf","page":"Home","title":"DecisionTreeAndRandomForest.Leaf","text":"Represents a Leaf in the ClassificationTree structure.\n\nvalues stores the labels of the data points.\n\n\n\n\n\n","category":"type"},{"location":"#DecisionTreeAndRandomForest.Node","page":"Home","title":"DecisionTreeAndRandomForest.Node","text":"Represents a Node in the ClassificationTree structure.\n\nleft points to the left child.\n\nright points to the right child.\n\nfeature_index stores the index of the selected feature.\n\nsplit_value stores the value on that the data is split.\n\ndata contains the datapoints of the Node.\n\nlabels contains the respective labels of the datapoints.\n\n\n\n\n\n","category":"type"},{"location":"#DecisionTreeAndRandomForest.best_split-Union{Tuple{L}, Tuple{T}, Tuple{AbstractMatrix{T}, Vector{L}}} where {T, L}","page":"Home","title":"DecisionTreeAndRandomForest.best_split","text":"best_split(X::AbstractMatrix{T}, y::Vector{T}) where T\n\nFind the best split for the dataset X and labels y based on Information Gain. Returns the best feature and threshold for the split.\n\nArguments\n\nX: A matrix of features.\ny: A vector of labels.\n\nReturns\n\nbest_feature: The index of the best feature to split on.\nbest_threshold: The threshold value for the best split.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.build_tree-Union{Tuple{L}, Tuple{T}, Tuple{Matrix{T}, Vector{L}, Int64, Int64}, Tuple{Matrix{T}, Vector{L}, Int64, Int64, Int64}} where {T, L}","page":"Home","title":"DecisionTreeAndRandomForest.build_tree","text":"build_tree(data, labels, max_depth, min_samples_split, depth)\n\nBuild the tree structure of the ClassificationTree\n\nIf depth is unspecified, it is set to 0\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.entropy-Union{Tuple{Vector{T}}, Tuple{T}} where T","page":"Home","title":"DecisionTreeAndRandomForest.entropy","text":"entropy(y::Vector{T}) where T\n\nCalculate the entropy of a vector of labels y.\n\nArguments\n\ny: A vector of labels.\n\nReturns\n\nThe entropy of the vector.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.find_best_split-Union{Tuple{L}, Tuple{T}, Tuple{Matrix{T}, Vector{L}}} where {T, L}","page":"Home","title":"DecisionTreeAndRandomForest.find_best_split","text":"Finds the best split point for a decision tree node. For now it uses the Gini impurity as splitting criterion, but should later be extended to support other criteria.\n\nArgs:     data: A matrix of features, where each row is a data point and each column is a feature.     labels: A vector of labels corresponding to the data points.\n\nReturns:     A tuple containing the index of the best feature and the best split value.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.fit-Tuple{ClassificationTree}","page":"Home","title":"DecisionTreeAndRandomForest.fit","text":"fit(tree::ClassificationTree)\n\nCompute the tree structure.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.gini_impurity-Union{Tuple{Vector{L}}, Tuple{L}} where L","page":"Home","title":"DecisionTreeAndRandomForest.gini_impurity","text":"Calculates the Gini impurity of a set of labels.\n\nArgs:     labels: A vector of labels.\n\nReturns:     The Gini impurity of the labels.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.information_gain-Union{Tuple{T}, Tuple{Vector{T}, Vector{T}, Vector{T}}} where T","page":"Home","title":"DecisionTreeAndRandomForest.information_gain","text":"information_gain(y::Vector{T}, y_left::Vector{T}, y_right::Vector{T}) where T\n\nCalculate the Information Gain of a split.\n\nArguments\n\ny: The original labels vector.\ny_left: The labels vector for the left split.\ny_right: The labels vector for the right split.\n\nReturns\n\nThe Information Gain of the split.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.predict-Union{Tuple{T}, Tuple{ClassificationTree, Matrix{T}}} where T","page":"Home","title":"DecisionTreeAndRandomForest.predict","text":"predict(tree::ClassificationTree, data::Matrix{T})\n\nReturns the prediction of the ClassificationTree for a list of datapoints.\n\ndata contains the datapoints to predict.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.print_tree-Tuple{ClassificationTree}","page":"Home","title":"DecisionTreeAndRandomForest.print_tree","text":"print_tree(tree:ClassificationTree)\n\nPrints the tree structure. Mainly used for debugging purposes.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.split_dataset-Union{Tuple{L}, Tuple{T}, Tuple{AbstractMatrix{T}, Vector{L}, Int64, Real}} where {T, L}","page":"Home","title":"DecisionTreeAndRandomForest.split_dataset","text":"split_dataset(X::AbstractMatrix{T}, y::Vector{T}, feature::Int, threshold::Real) where T\n\nSplit the dataset X and labels y based on a feature and a threshold. Returns the left and right splits for both X and y.\n\nArguments\n\nX: A matrix of features.\ny: A vector of labels.\nfeature: The index of the feature to split on.\nthreshold: The threshold value to split the feature.\n\nReturns\n\nX_left, y_left: The left split of the dataset and labels.\nX_right, y_right: The right split of the dataset and labels.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.split_node-Union{Tuple{L}, Tuple{T}, Tuple{Matrix{T}, Vector{L}, Any, Any}} where {T, L}","page":"Home","title":"DecisionTreeAndRandomForest.split_node","text":"Splits the labels into two nodes based on the provided feature and value.\n\nArgs:     data: A matrix of features, where each row is a data point and each column is a feature.     labels: A vector of labels corresponding to the data points.     index: The index of the feature to split on.     value: The value to split the feature on.\n\nReturns:     A tuple containing the left and right sets of labels.\n\n\n\n\n\n","category":"method"},{"location":"#DecisionTreeAndRandomForest.weighted_gini-Union{Tuple{L}, Tuple{Vector{L}, Vector{L}}} where L","page":"Home","title":"DecisionTreeAndRandomForest.weighted_gini","text":"Calculates the weighted Gini impurity of a split.\n\nArgs:     leftdataset: A vector of labels for the left subset of the data.     rightdataset: A vector of labels for the right subset of the data.\n\nReturns:     The weighted Gini impurity of the split.\n\n\n\n\n\n","category":"method"}]
}