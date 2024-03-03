struct MultiRegressionSplittingRule
    alpha::Float64
    imbalancePenalty::Float64
    numOutcomes::UInt
    counter::Vector{UInt}
    sums::Matrix{Float64}
    weightSums::Vector{Float64}
end

function MultiRegressionSplittingRule(maxNumUniqueValues::UInt, alpha::Float64, imbalancePenalty::Float64, numOutcomes::UInt)
    new(alpha, imbalancePenalty, numOutcomes, zeros(UInt, maxNumUniqueValues), zeros(Float64, maxNumUniqueValues, numOutcomes), zeros(Float64, maxNumUniqueValues))
end
function findBestSplit(splittingRule::MultiRegressionSplittingRule, data::Data, node::UInt, possibleSplitVars::Vector{UInt}, 
                       responsesBySample, samples::Vector{Vector{UInt}}, splitVars::Vector{UInt}, 
                       splitValues::Vector{Float64}, sendMissingLeft::Vector{Bool})
    sizeNode = length(samples[node])
    minChildSize = max(UInt(ceil(sizeNode * splittingRule.alpha)), 1)

    # Precompute sum of outcomes in this node
    sumNode = zeros(Float64, splittingRule.numOutcomes)
    weightSumNode = 0.0
    for sample in samples[node]
        sampleWeight = getWeight(data, sample)
        weightSumNode += sampleWeight
        sumNode .+= sampleWeight * responsesBySample[sample, :]
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
                           responsesBySample, samples, splittingRule.numOutcomes, splittingRule.imbalancePenalty)
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



function findBestSplitValue(data::Data, node::UInt, var::UInt, weightSumNode::Float64, sumNode, sizeNode::UInt, 
                            minChildSize::UInt, bestValue::Ref{Float64}, bestVar::Ref{UInt}, bestDecrease::Ref{Float64}, 
                            bestSendMissingLeft::Ref{Bool}, responsesBySample, samples::Vector{Vector{UInt}}, 
                            numOutcomes::UInt, imbalancePenalty::Float64)
    possibleSplitValues, sortedSamples = getAllValues(data, samples[node], var)

    if length(possibleSplitValues) < 2
        return
    end

    numSplits = length(possibleSplitValues) - 1
    weightSums = zeros(Float64, numSplits)
    counter = zeros(UInt, numSplits)
    sums = zeros(Float64, numSplits, numOutcomes)
    nMissing = 0
    weightSumMissing = 0.0
    sumMissing = zeros(Float64, numOutcomes)

    splitIndex = 1
    for i in 1:sizeNode - 1
        sample = sortedSamples[i]
        nextSample = sortedSamples[i + 1]
        sampleValue = get(data, sample, var)
        sampleWeight = getWeight(data, sample)

        if isnan(sampleValue)
            weightSumMissing += sampleWeight
            sumMissing += sampleWeight * responsesBySample[sample, :]
            nMissing += 1
        else
            weightSums[splitIndex] += sampleWeight
            sums[splitIndex, :] += sampleWeight * responsesBySample[sample, :]
            counter[splitIndex] += 1
        end

        nextSampleValue = get(data, nextSample, var)
        if sampleValue != nextSampleValue && !isnan(nextSampleValue)
            splitIndex += 1
        end
    end

    nLeft = nMissing
    weightSumLeft = weightSumMissing
    sumLeft = sumMissing

    for sendLeft in [true, false]
        if !sendLeft && nMissing == 0
            break
        end

        if !sendLeft
            nLeft = 0
            weightSumLeft = 0.0
            sumLeft .= 0.0
        end

        for i in 1:numSplits
            if i == 1 && !sendLeft
                continue
            end

            nLeft += counter[i]
            weightSumLeft += weightSums[i]
            sumLeft .+= sums[i, :]

            if nLeft < minChildSize
                continue
            end

            nRight = sizeNode - nLeft
            if nRight < minChildSize
                break
            end

            weightSumRight = weightSumNode - weightSumLeft
            decrease = sum(sumLeft .^ 2) / weightSumLeft + sum((sumNode - sumLeft) .^ 2) / weightSumRight
            penalty = imbalancePenalty * (1.0 / nLeft + 1.0 / nRight)
            decrease -= penalty

            if decrease > bestDecrease[]
                bestValue[] = possibleSplitValues[i]
                bestVar[] = var
                bestDecrease[] = decrease
                bestSendMissingLeft[] = sendLeft
            end
        end
    end
end
