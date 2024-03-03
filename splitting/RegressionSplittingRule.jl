struct RegressionSplittingRule
    alpha::Float64
    imbalancePenalty::Float64
    counter::Vector{UInt}
    sums::Vector{Float64}
    weightSums::Vector{Float64}
end

function RegressionSplittingRule(maxNumUniqueValues::UInt, alpha::Float64, imbalancePenalty::Float64)
    new(alpha, imbalancePenalty, zeros(UInt, maxNumUniqueValues), zeros(Float64, maxNumUniqueValues), zeros(Float64, maxNumUniqueValues))
end


function findBestSplit(splittingRule::RegressionSplittingRule, data::Data, node::UInt, possibleSplitVars::Vector{UInt}, 
                       responsesBySample, samples::Vector{Vector{UInt}}, splitVars::Vector{UInt}, 
                       splitValues::Vector{Float64}, sendMissingLeft::Vector{Bool})
    sizeNode = length(samples[node])
    minChildSize = max(UInt(ceil(sizeNode * splittingRule.alpha)), 1)

    # Precompute sum of outcomes in this node
    sumNode = 0.0
    weightSumNode = 0.0
    for sample in samples[node]
        sampleWeight = getWeight(data, sample)
        weightSumNode += sampleWeight
        sumNode += sampleWeight * responsesBySample[sample, 1] # Assuming the first column represents the response
    end

    # Initialize variables to track the best split variable
    bestVar = 0
    bestValue = 0.0
    bestDecrease = 0.0
    bestSendMissingLeft = true

    # For all possible split variables
    for var in possibleSplitVars
        findBestSplitValue(data, node, var, weightSumNode, sumNode, sizeNode, minChildSize, 
                           Ref(bestValue), Ref(bestVar), Ref(bestDecrease), Ref(bestSendMissingLeft), 
                           responsesBySample, samples) # Assuming findBestSplitValue is defined
    end

    # Stop if no good split found
    if bestDecrease <= 0.0
        return true
    end

    # Save best values
    splitVars[node] = bestVar
    splitValues[node] = bestValue
    sendMissingLeft[node] = bestSendMissingLeft
    return false
end
