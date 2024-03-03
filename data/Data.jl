

using Statistics
using LinearAlgebra

# Define the Data type to hold information about data set
struct Data
    dataPtr::Array{Float64, 2}
    numRows::Int
    numCols::Int
    outcomeIndex::Array{Int, 1}
    treatmentIndex::Array{Int, 1}
    instrumentIndex::Int
    weightIndex::Int
    causalSurvivalNumeratorIndex::Int
    causalSurvivalDenominatorIndex::Int
    censorIndex::Int
    disallowedSplitVariables::Set{Int}
end

# Constructor for Data type
function Data(dataPtr::Array{Float64, 2}, numRows::Int, numCols::Int)
    if dataPtr == nothing
        throw(ArgumentError("Invalid data storage: nullptr"))
    end

    return new(dataPtr, numRows, numCols, Int[], Int[], 0, 0, 0, 0, 0, Set{Int}())
end

# Alternative constructor using a vector
Data(data::Vector{Float64}, numRows::Int, numCols::Int) = Data(reshape(data, numRows, numCols), numRows, numCols)

# Alternative constructor using a pair of vectors
Data(data::Pair{Vector{Float64}, Vector{Int}}) = Data(data.first, data.second[1], data.second[2])

# Set outcome index
function setOutcomeIndex(data::Data, index::Int)
    setOutcomeIndex(data, [index])
end

function setOutcomeIndex(data::Data, index::Vector{Int})
    data.outcomeIndex = index
    union!(data.disallowedSplitVariables, index)
end

# Set treatment index
function setTreatmentIndex(data::Data, index::Int)
    setTreatmentIndex(data, [index])
end

function setTreatmentIndex(data::Data, index::Vector{Int})
    data.treatmentIndex = index
    union!(data.disallowedSplitVariables, index)
end

# Set instrument index
function setInstrumentIndex(data::Data, index::Int)
    data.instrumentIndex = index
    push!(data.disallowedSplitVariables, index)
end

# Set weight index
function setWeightIndex(data::Data, index::Int)
    data.weightIndex = index
    push!(data.disallowedSplitVariables, index)
end

# Set causal survival numerator index
function setCausalSurvivalNumeratorIndex(data::Data, index::Int)
    data.causalSurvivalNumeratorIndex = index
    push!(data.disallowedSplitVariables, index)
end

# Set causal survival denominator index
function setCausalSurvivalDenominatorIndex(data::Data, index::Int)
    data.causalSurvivalDenominatorIndex = index
    push!(data.disallowedSplitVariables, index)
end

# Set censor index
function setCensorIndex(data::Data, index::Int)
    data.censorIndex = index
    push!(data.disallowedSplitVariables, index)
end

# Get all values for a specific variable, sorted based on the samples
function getAllValues(data::Data, samples::Vector{Int}, var::Int)
    allValues = Float64[get(data, sample, var) for sample in samples]
    index = collect(1:length(samples))

    # Sort index based on split values (argsort)
    # NaNs are placed at the beginning
    sort!(index, by = i -> (allValues[i], isnan(allValues[i])), lt = isless)

    sortedSamples = [samples[idx] for idx in index]
    allValuesSorted = Float64[get(data, sample, var) for sample in sortedSamples]

    # Removing duplicates and handling NaNs
    uniqueValues = Float64[]
    lastValue = NaN
    for value in allValuesSorted
        if value ≠ lastValue && (!isnan(value) || isnan(lastValue))
            push!(uniqueValues, value)
            lastValue = value
        end
    end

    return uniqueValues, sortedSamples, index
end

# Function to retrieve the number of columns
getNumCols(data::Data) = data.numCols

# Function to retrieve the number of rows
getNumRows(data::Data) = data.numRows

# Function to retrieve the number of outcomes
function getNumOutcomes(data::Data)
    isempty(data.outcomeIndex) ? 1 : length(data.outcomeIndex)
end

# Function to retrieve the number of treatments
function getNumTreatments(data::Data)
    isempty(data.treatmentIndex) ? 1 : length(data.treatmentIndex)
end

# Function to retrieve the set of disallowed split variables
getDisallowedSplitVariables(data::Data) = data.disallowedSplitVariables

# Function to get a specific data value (not defined in the original snippet, but necessary for completeness)
function get(data::Data, row::Int, col::Int)
    # Assuming 1-based indexing in Julia, as opposed to 0-based in C++
    return data.dataPtr[row, col]
end
