using LinearAlgebra

struct LocalLinearPredictionStrategy
    lambdas::Vector{Float64}
    weightPenalty::Bool
    linearCorrectionVariables::Vector{UInt}
end

function LocalLinearPredictionStrategy(lambdas::Vector{Float64}, weightPenalty::Bool, linearCorrectionVariables::Vector{UInt})
    new(lambdas, weightPenalty, linearCorrectionVariables)
end

function predictionLength(strategy::LocalLinearPredictionStrategy)
    return length(strategy.lambdas)
end

function predict(sampleID::UInt, weightsBySampleID::Dict{UInt, Float64}, trainData::Data, data::Data, 
                 linearCorrectionVariables::Vector{UInt}, lambdas::Vector{Float64}, weightPenalty::Bool)
    numVariables = length(linearCorrectionVariables)
    numNonzeroWeights = length(weightsBySampleID)

    indices = UInt[]
    weightsVec = Float64[]

    for (index, weight) in weightsBySampleID
        push!(indices, index)
        push!(weightsVec, weight)
    end

    X = zeros(Float64, numNonzeroWeights, numVariables + 1)
    Y = zeros(Float64, numNonzeroWeights)

    for i in 1:numNonzeroWeights
        for j in 1:numVariables
            currentPredictor = linearCorrectionVariables[j]
            X[i, j + 1] = get(trainData, indices[i], currentPredictor) - get(data, sampleID, currentPredictor)
        end
        Y[i] = getOutcome(trainData, indices[i])
        X[i, 1] = 1
    end

    MUnpenalized = X' * Diagonal(weightsVec) * X
    predictions = Float64[]

    for lambda in lambdas
        M = MUnpenalized
        if !weightPenalty
            normalization = tr(M) / (numVariables + 1)
            for j in 2:numVariables + 1
                M[j, j] += lambda * normalization
            end
        else
            for j in 2:numVariables + 1
                M[j, j] += lambda * M[j, j]
            end
        end

        localCoefficients = M \ (X' * Diagonal(weightsVec) * Y)
        push!(predictions, localCoefficients[1])
    end

    return predictions
end

using LinearAlgebra

function computeVariance(sampleID::UInt, samplesByTree::Vector{Vector{UInt}}, weightsBySampleID::Dict{UInt, Float64}, 
                         trainData::Data, data::Data, ciGroupSize::UInt, linearCorrectionVariables::Vector{UInt}, 
                         lambda::Float64, weightPenalty::Bool, bayesDebiaser)
    numVariables = length(linearCorrectionVariables)
    numNonzeroWeights = length(weightsBySampleID)

    sampleIndexMap = Dict{UInt, UInt}()
    indices = UInt[]
    weightsVec = Float64[]

    for (index, weight) in weightsBySampleID
        sampleIndexMap[index] = length(indices) + 1
        push!(indices, index)
        push!(weightsVec, weight)
    end

    X = zeros(Float64, numNonzeroWeights, numVariables + 1)
    Y = zeros(Float64, numNonzeroWeights)

    for i in 1:numNonzeroWeights
        X[i, 1] = 1
        for j in 1:numVariables
            currentPredictor = linearCorrectionVariables[j]
            X[i, j + 1] = get(trainData, indices[i], currentPredictor) - get(data, sampleID, currentPredictor)
        end
        Y[i] = getOutcome(trainData, indices[i])
    end

    M = X' * Diagonal(weightsVec) * X
    if !weightPenalty
        normalization = tr(M) / (numVariables + 1)
        M[2:end, 2:end] .+= lambda * normalization
    else
        for i in 2:numVariables + 1
            M[i, i] += lambda * M[i, i]
        end
    end

    theta = M \ (X' * Diagonal(weightsVec) * Y)
    zeta = M \ [1; zeros(numVariables)]
    XTimesZeta = X * zeta
    localPrediction = X * theta
    pseudoResidual = XTimesZeta .* (Y - localPrediction)

    numGoodGroups = 0.0
    psiSquared = 0.0
    psiGroupedSquared = 0.0
    avgScore = 0.0

    for group in 1:length(samplesByTree) ÷ ciGroupSize
        goodGroup = all(j -> !isempty(samplesByTree[group * ciGroupSize + j]), 1:ciGroupSize)
        if !goodGroup
            continue
        end

        numGoodGroups += 1
        groupPsi = 0.0

        for j in 1:ciGroupSize
            b = group * ciGroupSize + j
            psi1 = sum(pseudoResidual[sampleIndexMap[sample]] for sample in samplesByTree[b]) / length(samplesByTree[b])
            psiSquared += psi1^2
            groupPsi += psi1
        end

        groupPsi /= ciGroupSize
        psiGroupedSquared += groupPsi^2
        avgScore += groupPsi
    end

    avgScore /= numGoodGroups
    varBetween = psiGroupedSquared / numGoodGroups - avgScore^2
    varTotal = psiSquared / (numGoodGroups * ciGroupSize) - avgScore^2
    groupNoise = (varTotal - varBetween) / (ciGroupSize - 1)

    varDebiased = bayesDebiaser.debias(varBetween, groupNoise, numGoodGroups) # Placeholder for bayes_debiaser.debias

    return [varDebiased]
end
