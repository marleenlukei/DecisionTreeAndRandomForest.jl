# Splitting Criterion

This package offers multiple options of splitting criterion for evaluating the quality of a split and therefore constructing a decision tree. In the following a brief overview of the available criterion and their use cases is provided.

## Gini Impurity
Gini Impurity measures the likelihood of an incorrect classification of a randomly chosen element if it is labeled according to the distribution of labels in the dataset. 
Gini Impurity is primarily used in classification problems. A lower Gini impurity indicates a purer node with a higher confidence in predicting the class label. The goal is to minimize the Gini Impurity at each split, thereby creating nodes that are as pure as possible.

``$$Gini(S) = 1 - \sum_{i=1}^{C} (p_i)^2$$``

where:
- ``S`` is the set of data points in the current node.\
- ``C`` is the number of classes.\
- ``p_i`` is the proportion of data points belonging to class ``i`` in ``S``.\


Steps for Calculation:
1. Calculate Gini coefficients for each child node.
2. Compute the impurity for each split using a weighted Gini score.
3. Choose the split with the lowest Gini impurity.


## Information Gain
Information Gain calculated as the difference in entropy before and after splitting a dataset on an attribute. Entropy measures the uncertainty or impurity in the data. The goal is to reduce entropy and maximize information gain, leading to a more informative split. Information Gain is used in classification problems to choose the attribute that provides the highest information gain, resulting in the most informative split.

The information gain for a dataset 𝑆 after a split on attribute 𝐴 is given by:

``Gain(S, A) = Entropy(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} Entropy(S_v)``\


where:
- ``𝐴`` is a feature of the dataset.\
- ``v`` is a specific value of the feature 𝐴.\
    \
Entropy is calculated as:

``Entropy(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)``


where:
- ``S`` is the set of data points in the current node.\
- ``C`` is the number of classes.\
- ``p_i`` is the proportion of data points belonging to class ``i`` in ``S``.\
\
Steps for Calculation:
1. Calculate the entropy of the original dataset 𝑆.
2. For each split on attribute 𝐴 calculate the entropy of each child node ``𝑆_𝑣`` and calculate the weighted entropy after the split.
3. Compute the Information Gain by subtracting the weighted entropy from the original entropy.
5. Choose the split with the highest information gain.


## Variance Reduction
Variance Reduction measures the reduction in variance of the target variable achieved by splitting a node. Higher variance reduction indicates a more informative split. Variance reduction is particularly useful for regression problems where the goal is to predict a continuous target variable.
 
``\text{VR}(S) = \sigma^2(S) - \sum_{i=1}^{n} \frac{|S_i|}{|S|} \sigma^2(S_i)``


where:
- ``\sigma^2(S)``: Variance of the parent node S.
- ``S``: Set of data points in the current node.
-  ``S_i``: Subsets of ``S`` after the split.
- ``n``: Number of subsets after the split.

Steps for Calculation:
1. Calculate the variance of the parent node ``S``.
2. For each child node ``S_i``, calculate its variance.
3. Compute the weighted sum of the variances of the child nodes ``S_i``.
4. Subtract the weighted sum from the variance of the parent node ``S`` to get the variance reduction.
5. Choose the split with the highest variance reduction.