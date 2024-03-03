struct TreeOptions
    mtry::UInt
    minNodeSize::UInt
    honesty::Bool
    honestyFraction::Float64
    honestyPruneLeaves::Bool
    alpha::Float64
    imbalancePenalty::Float64
end

function TreeOptions(mtry::UInt, minNodeSize::UInt, honesty::Bool, honestyFraction::Float64, 
                     honestyPruneLeaves::Bool, alpha::Float64, imbalancePenalty::Float64)
    new(mtry, minNodeSize, honesty, honestyFraction, honestyPruneLeaves, alpha, imbalancePenalty)
end

getMtry(options::TreeOptions) = options.mtry

getMinNodeSize(options::TreeOptions) = options.minNodeSize

getHonesty(options::TreeOptions) = options.honesty

getHonestyFraction(options::TreeOptions) = options.honestyFraction

getHonestyPruneLeaves(options::TreeOptions) = options.honestyPruneLeaves

getAlpha(options::TreeOptions) = options.alpha

getImbalancePenalty(options::TreeOptions) = options.imbalancePenalty
