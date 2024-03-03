struct SplitFrequencyComputer end

function compute(splitFreqComputer::SplitFrequencyComputer, forest::Forest, maxDepth::Int)
    numVariables = getNumVariables(forest)
    result = [zeros(Int, numVariables) for _ in 1:maxDepth]

    for tree in getTrees(forest)
        childNodes = getChildNodes(tree)
        level = [getRootNode(tree)]
        depth = 1

        while !isempty(level) && depth <= maxDepth
            nextLevel = []

            for node in level
                if isLeaf(tree, node)
                    continue
                end

                variable = getSplitVars(tree)[node]
                result[depth][variable] += 1

                push!(nextLevel, childNodes[1][node])
                push!(nextLevel, childNodes[2][node])
            end

            level = nextLevel
            depth += 1
        end
    end

    return result
end
