using DecisionTreeAndRandomForest
using Test

@testset "RandomForest" begin
    
    data = ["dog" 37.0; "dog" 38.4; "dog" 40.2; "dog" 38.9; "human" 36.2; "human" 37.4; "human" 38.8; "human" 36.2]
    labels = ["healthy", "healthy", "sick", "healthy", "healthy", "sick", "sick", "healthy"]
    rf = RandomForest(3, -1, 2, 2, data, labels)
    fit_forest(rf)
    test_data = ["dog" 38.0; "human" 38.0]
    prediction = predict_forest(rf, test_data)
    # println("Predictions: ", prediction)
    @test prediction[1] == "healthy"
    @test prediction[2] == "sick"


end
