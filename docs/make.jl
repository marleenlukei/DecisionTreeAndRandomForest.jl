using DecisionTreeAndRandomForest
using Documenter

DocMeta.setdocmeta!(DecisionTreeAndRandomForest, :DocTestSetup, :(using DecisionTreeAndRandomForest); recursive=true)

makedocs(;
    modules=[DecisionTreeAndRandomForest],
    authors="Marleen Lukei marleen.lukei@campus.tu-berlin.de>",
    sitename="DecisionTreeAndRandomForest.jl",
    format=Documenter.HTML(;
        canonical="https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
)

deploydocs(;
    repo="github.com/marleenlukei/DecisionTreeAndRandomForest.jl",
    devbranch="master",
)
