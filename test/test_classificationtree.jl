using DecisionTreeAndRandomForest
using Test

@testset "ClassificationTree" begin
    data = [0 37.0; 0 38.4; 0 40.2; 0 38.9; 1 36.2; 1 37.4; 1 38.8; 1 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
    tree = ClassificationTree(data, labels)
    fit(tree)
    test_data = [0 38.0; 1 38.0]
    prediction = predict(tree, test_data)
    print_tree(tree)
    @test prediction[1] == "healthy"
    @test prediction[2] == "sick"
end