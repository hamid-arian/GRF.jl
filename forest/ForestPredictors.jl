function instrumentalPredictor(numThreads::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = InstrumentalPredictionStrategy()
    return ForestPredictor(numThreads, predictionStrategy)
end

function multiCausalPredictor(numThreads::UInt, numTreatments::UInt, numOutcomes::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = MultiCausalPredictionStrategy(numTreatments, numOutcomes)
    return ForestPredictor(numThreads, predictionStrategy)
end

function quantilePredictor(numThreads::UInt, quantiles::Vector{Float64})
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = QuantilePredictionStrategy(quantiles)
    return ForestPredictor(numThreads, predictionStrategy)
end

function probabilityPredictor(numThreads::UInt, numClasses::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = ProbabilityPredictionStrategy(numClasses)
    return ForestPredictor(numThreads, predictionStrategy)
end

function regressionPredictor(numThreads::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = RegressionPredictionStrategy()
    return ForestPredictor(numThreads, predictionStrategy)
end

function multiRegressionPredictor(numThreads::UInt, numOutcomes::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = MultiRegressionPredictionStrategy(numOutcomes)
    return ForestPredictor(numThreads, predictionStrategy)
end

function llRegressionPredictor(numThreads::UInt, lambdas::Vector{Float64}, weightPenalty::Bool, linearCorrectionVariables::Vector{UInt})
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = LocalLinearPredictionStrategy(lambdas, weightPenalty, linearCorrectionVariables)
    return ForestPredictor(numThreads, predictionStrategy)
end

function llCausalPredictor(numThreads::UInt, lambdas::Vector{Float64}, weightPenalty::Bool, linearCorrectionVariables::Vector{UInt})
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = LLCausalPredictionStrategy(lambdas, weightPenalty, linearCorrectionVariables)
    return ForestPredictor(numThreads, predictionStrategy)
end

function survivalPredictor(numThreads::UInt, numFailures::UInt, predictionType::Int)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = SurvivalPredictionStrategy(numFailures, predictionType)
    return ForestPredictor(numThreads, predictionStrategy)
end

function causalSurvivalPredictor(numThreads::UInt)
    numThreads = validateNumThreads(numThreads)
    predictionStrategy = CausalSurvivalPredictionStrategy()
    return ForestPredictor(numThreads, predictionStrategy)
end
