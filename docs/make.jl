using Documenter
using DecisionTreeAndRandomForest

DocMeta.setdocmeta!(DecisionTreeAndRandomForest, :DocTestSetup, :(using DecisionTreeAndRandomForest); recursive=true)

makedocs(
    modules=[DecisionTreeAndRandomForest],
    authors="Marleen Lukei <marleen.lukei@campus.tu-berlin.de>, Cedric Lenßen <c.lenssen@campus.tu-berlin.de>, Arkadeep Dutta <arkadeep.dutta@campus.tu-berlin.de>, Enea Gurra <e.gurra@campus.tu-berlin.de>",
    sitename="DecisionTreeAndRandomForest.jl",
    format=Documenter.HTML(;
        canonical="https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Splitting Criterion" => "splitting_criterion.md",
        "Dataset Examples" => "dataset_examples.md",
    ],
)

deploydocs(;
    repo="github.com/marleenlukei/DecisionTreeAndRandomForest.jl",
    devbranch="main",
)
