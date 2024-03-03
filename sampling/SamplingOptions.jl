struct SamplingOptions
    numSamplesPerCluster::UInt
    clusters::Vector{Vector{UInt}}
end

function SamplingOptions()
    new(0, Vector{UInt}[])
end

function SamplingOptions(samplesPerCluster::UInt, sampleClusters::Vector{UInt})
    clusterIds = Dict{UInt, UInt}()
    for cluster in sampleClusters
        if !haskey(clusterIds, cluster)
            clusterIds[cluster] = length(clusterIds) + 1
        end
    end

    clusters = [UInt[] for _ in 1:length(clusterIds)]
    for sample in 1:length(sampleClusters)
        cluster = sampleClusters[sample]
        clusterId = clusterIds[cluster]
        push!(clusters[clusterId], sample)
    end

    new(samplesPerCluster, clusters)
end

function getSamplesPerCluster(options::SamplingOptions)
    options.numSamplesPerCluster
end

function getClusters(options::SamplingOptions)
    options.clusters
end
