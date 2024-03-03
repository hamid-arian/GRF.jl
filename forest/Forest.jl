include("../tree/Tree.jl")


struct Forest
    trees::Vector{Tree}
    numVariables::UInt
    ciGroupSize::UInt
end

function Forest(trees::Vector{Tree}, numVariables::UInt, ciGroupSize::UInt)
    new(trees, numVariables, ciGroupSize)
end

#形成森林,将多个 Forest 实例合并为一个的函数
function mergeForests(forests::Vector{Forest})
    allTrees = Tree[]
    numVariables = forests[1].numVariables
    ciGroupSize = forests[1].ciGroupSize

    for forest in forests
        append!(allTrees, forest.trees)

        if forest.ciGroupSize != ciGroupSize
            throw(ArgumentError("All forests being merged must have the same ci_group_size."))
        end
    end

    return Forest(allTrees, numVariables, ciGroupSize)
end

getTrees(forest::Forest) = forest.trees
getNumVariables(forest::Forest) = forest.numVariables
getCiGroupSize(forest::Forest) = forest.ciGroupSize
