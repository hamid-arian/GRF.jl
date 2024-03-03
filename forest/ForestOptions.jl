include("../tree/TreeOptions.jl")
include("../sampling/SamplingOptions.jl")

struct ForestOptions
    numTrees::UInt
    ciGroupSize::UInt
    sampleFraction::Float64
    treeOptions::TreeOptions
    samplingOptions::SamplingOptions
    numThreads::UInt
    randomSeed::UInt
end

function ForestOptions(numTrees::UInt, ciGroupSize::UInt, sampleFraction::Float64, mtry::UInt, 
                       minNodeSize::UInt, honesty::Bool, honestyFraction::Float64, honestyPruneLeaves::Bool, 
                       alpha::Float64, imbalancePenalty::Float64, numThreads::UInt, randomSeed::UInt, 
                       sampleClusters::Vector{UInt}, samplesPerCluster::UInt)
    treeOptions = TreeOptions(mtry, minNodeSize, honesty, honestyFraction, honestyPruneLeaves, alpha, imbalancePenalty)
    samplingOptions = SamplingOptions(samplesPerCluster, sampleClusters)

    validatedNumThreads = validateNumThreads(numThreads)
    adjustedNumTrees = numTrees + (numTrees % ciGroupSize)

    if ciGroupSize > 1 && sampleFraction > 0.5
        throw(ArgumentError("When confidence intervals are enabled, the sampling fraction must be less than 0.5."))
    end

    new(adjustedNumTrees, ciGroupSize, sampleFraction, treeOptions, samplingOptions, validatedNumThreads, randomSeed)
end

function validateNumThreads(numThreads::UInt)
    if numThreads == 0 # Assuming 0 represents DEFAULT_NUM_THREADS
        return Sys.CPU_THREADS
    elseif numThreads > 0
        return numThreads
    else
        throw(ArgumentError("A negative number of threads was provided."))
    end
end

# Getter for the number of trees
getNumTrees(options::ForestOptions) = options.numTrees

# Getter for the confidence interval group size
getCiGroupSize(options::ForestOptions) = options.ciGroupSize

# Getter for the sample fraction
getSampleFraction(options::ForestOptions) = options.sampleFraction

# Getter for the tree options
getTreeOptions(options::ForestOptions) = options.treeOptions

# Getter for the sampling options
getSamplingOptions(options::ForestOptions) = options.samplingOptions

# Getter for the number of threads
getNumThreads(options::ForestOptions) = options.numThreads

# Getter for the random seed
getRandomSeed(options::ForestOptions) = options.randomSeed
