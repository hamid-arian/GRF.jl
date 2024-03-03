include("../data/Data.jl")

# 定义结构体，使用正确的命名约定并避免类型循环引用
struct PredictionValues
    values::Vector{Float64}  # 假设我们存储的是一个浮点数数组
end



struct Tree
    rootNode::UInt
    childNodes::Vector{Vector{UInt}}
    leafSamples::Vector{Vector{UInt}}
    splitVars::Vector{UInt}
    splitValues::Vector{Float64}
    drawnSamples::Vector{UInt}
    sendMissingLeft::Vector{Bool}
    predictionValues::PredictionValues
end


function Tree(rootNode::UInt, childNodes::Vector{Vector{UInt}}, leafSamples::Vector{Vector{UInt}}, 
              splitVars::Vector{UInt}, splitValues::Vector{Float64}, drawnSamples::Vector{UInt},
              sendMissingLeft::Vector{Bool}, predictionValues::PredictionValues)
    new(rootNode, childNodes, leafSamples, splitVars, splitValues, drawnSamples, 
        sendMissingLeft, predictionValues)
end

getRootNode(tree::Tree) = tree.rootNode

getChildNodes(tree::Tree) = tree.childNodes

getLeafSamples(tree::Tree) = tree.leafSamples

getSplitVars(tree::Tree) = tree.splitVars

getSplitValues(tree::Tree) = tree.splitValues

getDrawnSamples(tree::Tree) = tree.drawnSamples

getSendMissingLeft(tree::Tree) = tree.sendMissingLeft

getPredictionValues(tree::Tree) = tree.predictionValues

function findLeafNodes(tree::Tree, data::Data, samples::Vector{UInt})
    predictionLeafNodes = UInt[]
    for sample in samples
        node = findLeafNode(tree, data, sample) # Assuming findLeafNode is defined elsewhere
        push!(predictionLeafNodes, node)
    end
    return predictionLeafNodes
end

function findLeafNodes(tree::Tree, data::Data, validSamples::Vector{Bool})
    numSamples = getNumRows(data) # Assuming getNumRows is defined elsewhere

    predictionLeafNodes = UInt[]
    for sample in 1:numSamples
        if !validSamples[sample]
            continue
        end

        node = findLeafNode(tree, data, sample) # Assuming findLeafNode is defined elsewhere
        push!(predictionLeafNodes, node)
    end
    return predictionLeafNodes
end

setLeafSamples(tree::Tree, leafSamples::Vector{Vector{UInt}}) = tree.leafSamples = leafSamples

setPredictionValues(tree::Tree, predictionValues::PredictionValues) = tree.predictionValues = predictionValues

# Method to find the leaf node for a given sample
function findLeafNode(tree::Tree, data::Data, sample::UInt)
    node = tree.rootNode
    while true
        if isLeaf(tree, node)
            break
        end

        splitVar = tree.splitVars[node]
        splitVal = tree.splitValues[node]
        value = get(data, sample, splitVar) # Assuming get is defined in Data
        sendNaLeft = tree.sendMissingLeft[node]
        if value <= splitVal || (sendNaLeft && isnan(value)) || (isnan(splitVal) && isnan(value))
            node = tree.childNodes[1][node]
        else
            node = tree.childNodes[2][node]
        end
    end
    return node
end

# Method to prune the tree based on honesty principle
function honestyPruneLeaves(tree::Tree)
    numNodes = length(tree.leafSamples)
    for n in numNodes:-1:tree.rootNode+1
        node = n - 1
        if isLeaf(tree, node)
            continue
        end

        leftChild = tree.childNodes[1][node]
        if !isLeaf(tree, leftChild)
            pruneNode(tree, leftChild)
        end

        rightChild = tree.childNodes[2][node]
        if !isLeaf(tree, rightChild)
            pruneNode(tree, rightChild)
        end
    end
    pruneNode(tree, tree.rootNode)
end

# Helper function to prune a node
function pruneNode(tree::Tree, node::UInt)
    leftChild = tree.childNodes[1][node]
    rightChild = tree.childNodes[2][node]

    if isEmptyLeaf(tree, leftChild) || isEmptyLeaf(tree, rightChild)
        tree.childNodes[1][node] = 0
        tree.childNodes[2][node] = 0

        if !isEmptyLeaf(tree, leftChild)
            node = leftChild
        elseif !isEmptyLeaf(tree, rightChild)
            node = rightChild
        end
    end
end

# Check if a node is a leaf
isLeaf(tree::Tree, node::UInt) = tree.childNodes[1][node] == 0 && tree.childNodes[2][node] == 0

# Check if a leaf node is empty
isEmptyLeaf(tree::Tree, node::UInt) = isLeaf(tree, node) && isempty(tree.leafSamples[node])


