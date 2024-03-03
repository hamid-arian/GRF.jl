include("../tree/TreeTrainer.jl")



struct TreeTraverser
    treetraverser::Vector{Float64} 
end

struct PredictionCollector
    predictionCollector::Vector{Float64} 
end

struct DefaultPredictionStrategy
    defaultPredictionStrategy::Vector{Float64} 
end

struct OptimizedPredictionStrategy
    optimizedPredictionStrategy::Vector{Float64} 
end

struct ForestPredictor
    treeTraverser::TreeTraverser
    predictionCollector::PredictionCollector
end

function ForestPredictor(numThreads::UInt, strategy::DefaultPredictionStrategy)
    treeTraverser = TreeTraverser(numThreads)
    predictionCollector = DefaultPredictionCollector(strategy, numThreads)
    new(treeTraverser, predictionCollector)
end

function ForestPredictor(numThreads::UInt, strategy::OptimizedPredictionStrategy)
    treeTraverser = TreeTraverser(numThreads)
    predictionCollector = OptimizedPredictionCollector(strategy, numThreads)
    new(treeTraverser, predictionCollector)
end
function predict(predictor::ForestPredictor, forest::Forest, trainData::Data, data::Data, 
                 estimateVariance::Bool; oobPrediction::Bool = false)
    if estimateVariance && getCiGroupSize(forest) <= 1
        throw(ArgumentError("To estimate variance during prediction, the forest must be trained with ci_group_size greater than 1."))
    end

    leafNodesByTree = getLeafNodes(predictor.treeTraverser, forest, data, oobPrediction)
    treesBySample = getValidTreesBySample(predictor.treeTraverser, forest, data, oobPrediction)

    return collectPredictions(predictor.predictionCollector, forest, trainData, data, 
                              leafNodesByTree, treesBySample, estimateVariance, oobPrediction)
end

function predictOOB(predictor::ForestPredictor, forest::Forest, data::Data, estimateVariance::Bool)
    return predict(predictor, forest, data, data, estimateVariance, oobPrediction = true)
end
