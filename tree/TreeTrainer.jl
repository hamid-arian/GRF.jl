include("../sampling/RandomSampler.jl")


struct TreeTrainer
    relabelingStrategy
    splittingRuleFactory
    predictionStrategy
end

function TreeTrainer(relabelingStrategy, splittingRuleFactory, predictionStrategy)
    new(relabelingStrategy, splittingRuleFactory, predictionStrategy)
end


function train(trainer::TreeTrainer, data::Data, sampler::RandomSampler, clusters::Vector{UInt}, options::TreeOptions)
    childNodes = [UInt[], UInt[]]
    nodes = [UInt[]]
    splitVars = UInt[]
    splitValues = Float64[]
    sendMissingLeft = Bool[]

    createEmptyNode(childNodes, nodes, splitVars, splitValues, sendMissingLeft)

    newLeafSamples = UInt[]

    if getHonesty(options)
        treeGrowingClusters, newLeafClusters = subsample(sampler, clusters, getHonestyFraction(options), true)
        sampleFromClusters(sampler, treeGrowingClusters, nodes[1])
        sampleFromClusters(sampler, newLeafClusters, newLeafSamples)
    else
        sampleFromClusters(sampler, clusters, nodes[1])
    end

    splittingRule = createSplittingRule(trainer.splittingRuleFactory, length(nodes[1]), options) # Placeholder for createSplittingRule

    numOpenNodes = 1
    i = 1
    responsesBySample = Array{Float64,2}(undef, getNumRows(data), getResponseLength(trainer.relabelingStrategy)) # Placeholder for getResponseLength

    while numOpenNodes > 0
        isLeafNode = splitNode(i, data, splittingRule, sampler, childNodes, nodes, splitVars, splitValues, sendMissingLeft, responsesBySample, options) # Placeholder for splitNode
        if isLeafNode
            numOpenNodes -= 1
        else
            empty!(nodes[i])
            numOpenNodes += 1
        end
        i += 1
    end

    drawnSamples = getSamplesInClusters(sampler, clusters)

    tree = Tree(0, childNodes, nodes, splitVars, splitValues, drawnSamples, sendMissingLeft, PredictionValues()) # Assuming PredictionValues is defined

    if !isempty(newLeafSamples)
        repopulateLeafNodes(tree, data, newLeafSamples, getHonestyPruneLeaves(options)) # Placeholder for repopulateLeafNodes
    end

    predictionValues = if trainer.predictionStrategy !== nothing
        precomputePredictionValues(trainer.predictionStrategy, getLeafSamples(tree), data) # Placeholder for precomputePredictionValues
    else
        PredictionValues() # Assuming PredictionValues is defined
    end
    setPredictionValues(tree, predictionValues)

    return tree
end



function repopulateLeafNodes(tree::Tree, data::Data, leafSamples::Vector{UInt}, honestyPruneLeaves::Bool)
    numNodes = length(getLeafSamples(tree))
    newLeafNodes = [UInt[] for _ in 1:numNodes]

    leafNodes = findLeafNodes(tree, data, leafSamples)

    for sample in leafSamples
        leafNode = leafNodes[sample]
        push!(newLeafNodes[leafNode], sample)
    end

    setLeafSamples(tree, newLeafNodes)
    
    if honestyPruneLeaves
        honestyPruneLeaves(tree)
    end
end
function createSplitVariableSubset(result::Vector{UInt}, sampler::RandomSampler, data::Data, mtry::UInt)
    numIndependentVariables = getNumCols(data) - length(getDisallowedSplitVariables(data))
    mtrySample = samplePoisson(sampler, mtry)
    splitMtry = max(min(mtrySample, numIndependentVariables), 1)

    draw(sampler, result, getNumCols(data), getDisallowedSplitVariables(data), splitMtry)
end

function splitNode(node::UInt, data::Data, splittingRule, sampler::RandomSampler, childNodes::Vector{Vector{UInt}}, 
                   samples::Vector{Vector{UInt}}, splitVars::Vector{UInt}, splitValues::Vector{Float64}, 
                   sendMissingLeft::Vector{Bool}, responsesBySample, options::TreeOptions)
    possibleSplitVars = UInt[]
    createSplitVariableSubset(possibleSplitVars, sampler, data, getMtry(options))

    stop = splitNodeInternal(node, data, splittingRule, possibleSplitVars, samples, splitVars, splitValues, sendMissingLeft, responsesBySample, getMinNodeSize(options)) # Placeholder for splitNodeInternal

    if stop
        return true
    end

    splitVar = splitVars[node]
    splitValue = splitValues[node]
    sendNaLeft = sendMissingLeft[node]

    leftChildNode = length(samples) + 1
    childNodes[1][node] = leftChildNode
    createEmptyNode(childNodes, samples, splitVars, splitValues, sendMissingLeft) # Placeholder for createEmptyNode

    rightChildNode = length(samples) + 1
    childNodes[2][node] = rightChildNode
    createEmptyNode(childNodes, samples, splitVars, splitValues, sendMissingLeft) # Placeholder for createEmptyNode

    # Distribute samples to left or right child based on split
    for sample in samples[node]
        value = get(data, sample, splitVar) # Assuming get is defined in Data
        if value <= splitValue || (sendNaLeft && isnan(value)) || (isnan(splitValue) && isnan(value))
            push!(samples[leftChildNode], sample)
        else
            push!(samples[rightChildNode], sample)
        end
    end

    # No terminal node
    return false
end

function splitNodeInternal(node::UInt, data::Data, splittingRule, possibleSplitVars::Vector{UInt}, 
                           samples::Vector{Vector{UInt}}, splitVars::Vector{UInt}, splitValues::Vector{Float64}, 
                           sendMissingLeft::Vector{Bool}, responsesBySample, minNodeSize::UInt)
    # Check node size, stop if minimum size reached
    if length(samples[node]) <= minNodeSize
        splitValues[node] = -1.0
        return true
    end

    stop = relabelingStrategy.relabel(samples[node], data, responsesBySample) # Placeholder for relabeling strategy

    if stop || findBestSplit(splittingRule, data, node, possibleSplitVars, responsesBySample, samples, splitVars, splitValues, sendMissingLeft) # Placeholder for findBestSplit
        splitValues[node] = -1.0
        return true
    end

    return false
end

function createEmptyNode(childNodes::Vector{Vector{UInt}}, samples::Vector{Vector{UInt}}, 
                         splitVars::Vector{UInt}, splitValues::Vector{Float64}, 
                         sendMissingLeft::Vector{Bool})
    push!(childNodes[1], 0) # Add a placeholder for the left child
    push!(childNodes[2], 0) # Add a placeholder for the right child
    push!(samples, UInt[])  # Add an empty vector for samples in the new node
    push!(splitVars, 0)     # Add a default split variable
    push!(splitValues, 0.0) # Add a default split value
    push!(sendMissingLeft, true) # By default, send missing values to the left
end
