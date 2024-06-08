using Documenter
using DecisionTreeAndRandomForest

DocMeta.setdocmeta!(DecisionTreeAndRandomForest, :DocTestSetup, :(using DecisionTreeAndRandomForest); recursive=true)

makedocs(
    sitename="DecisionTreeAndRandomForest.jl",
    authors="Marleen Lukei <marleen.lukei@campus.tu-berlin.de>, Cedric Lenßen <c.lenssen@campus.tu-berlin.de>, Arkadeep Dutta, Enea Gurra",
    modules=[DecisionTreeAndRandomForest],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
    format=Documenter.HTML(;
        canonical="https://marleenlukei.github.io/DecisionTreeAndRandomForest.jl",
        edit_link="main",
        assets=String[],
    ),
)

deploydocs(;
    repo="github.com/marleenlukei/DecisionTreeAndRandomForest.jl",
    devbranch="main",
)