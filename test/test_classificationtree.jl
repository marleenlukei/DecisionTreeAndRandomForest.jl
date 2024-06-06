using DecisionTreeAndRandomForest
using Test

@testset "ClassificationTree" begin
    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
    # data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    # labels = ["healthy", "healthy", "healthy", "healthy", "sick", "sick", "sick", "sick"]
    tree = ClassificationTree(data, labels)
    fit(tree)
    test_data = ["dog" 38.0; "human" 38.0]
    prediction = predict(tree, test_data)
    print_tree(tree)
    @test prediction[1] == "healthy"
    @test prediction[2] == "sick"
end