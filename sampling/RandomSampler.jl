

using Random

struct RandomSampler
    seed::UInt
    options::SamplingOptions
    rng::MersenneTwister
end

function RandomSampler(seed::UInt, options::SamplingOptions)
    rng = MersenneTwister(seed)
    new(seed, options, rng)
end

function sampleClusters(sampler::RandomSampler, numRows::UInt, sampleFraction::Float64)
    clusters = getClusters(sampler.options)

    if isempty(clusters)
        return sample(sampler, numRows, sampleFraction)
    else
        numSamples = length(clusters)
        return sample(sampler, numSamples, sampleFraction)
    end
end

function sample(sampler::RandomSampler, numSamples::UInt, sampleFraction::Float64)
    numSamplesInbag = UInt(floor(numSamples * sampleFraction))
    samples = shuffle(1:numSamples, sampler.rng)
    return samples[1:numSamplesInbag]
end

function subsample(sampler::RandomSampler, samples::Vector{UInt}, sampleFraction::Float64)
    shuffledSamples = shuffle(samples, sampler.rng)
    subsampleSize = UInt(ceil(length(samples) * sampleFraction))
    return shuffledSamples[1:subsampleSize]
end

function subsample(sampler::RandomSampler, samples::Vector{UInt}, sampleFraction::Float64, outOfBag::Bool)
    shuffledSamples = shuffle(samples, sampler.rng)
    subsampleSize = UInt(ceil(length(samples) * sampleFraction))
    subsamples = shuffledSamples[1:subsampleSize]
    oobSamples = shuffledSamples[subsampleSize + 1:end]
    return subsamples, oobSamples
end

function subsampleWithSize(sampler::RandomSampler, samples::Vector{UInt}, subsampleSize::UInt)
    shuffledSamples = shuffle(samples, sampler.rng)
    return shuffledSamples[1:subsampleSize]
end

function sampleFromClusters(sampler::RandomSampler, clusters::Vector{UInt})
    clusterOptions = getClusters(sampler.options)

    if isempty(clusterOptions)
        return clusters
    else
        samples = UInt[]
        for cluster in clusters
            clusterSamples = clusterOptions[cluster]
            if length(clusterSamples) <= getSamplesPerCluster(sampler.options)
                append!(samples, clusterSamples)
            else
                subsamples = subsampleWithSize(sampler, clusterSamples, getSamplesPerCluster(sampler.options))
                append!(samples, subsamples)
            end
        end
        return samples
    end
end

using Random


# Get samples in clusters
function getSamplesInClusters(sampler::RandomSampler, clusters::Vector{UInt})
    samples = UInt[]
    if isempty(getClusters(sampler.options))
        samples = clusters
    else
        for cluster in clusters
            clusterSamples = getClusters(sampler.options)[cluster]
            append!(samples, clusterSamples)
        end
    end
    return samples
end

# Shuffle and split a sequence of numbers
function shuffleAndSplit(sampler::RandomSampler, nAll::UInt, size::UInt)
    samples = shuffle(0:(nAll - 1), sampler.rng)
    return samples[1:size]
end

# Draw samples with two methods depending on the number of samples
function draw(sampler::RandomSampler, max::UInt, skip::Set{UInt}, numSamples::UInt)
    if numSamples < max / 10
        return drawSimple(sampler, max, skip, numSamples)
    else
        return drawFisherYates(sampler, max, skip, numSamples)
    end
end

# Simple draw method
function drawSimple(sampler::RandomSampler, max::UInt, skip::Set{UInt}, numSamples::UInt)
    result = UInt[]
    temp = falses(max)

    for _ in 1:numSamples
        draw = rand(0:max-1-length(skip))
        for skipValue in skip
            if draw >= skipValue
                draw += 1
            end
        end

        while temp[draw + 1]
            draw = rand(0:max-1-length(skip))
        end

        temp[draw + 1] = true
        push!(result, draw)
    end
    return result
end

# Fisher-Yates draw method
function drawFisherYates(sampler::RandomSampler, max::UInt, skip::Set{UInt}, numSamples::UInt)
    result = collect(0:(max - 1))
    for skipValue in skip
        deleteat!(result, skipValue + 1)
    end

    for i in 1:numSamples
        j = i + rand(0:(max-length(skip)-i))
        result[i], result[j] = result[j], result[i]
    end

    return result[1:numSamples]
end

# Sample a Poisson distribution
function samplePoisson(sampler::RandomSampler, mean::UInt)
    return rand(Poisson(mean))
end


