# 定义一个线性回归重新标记策略的结构体Structural body
struct LLRegressionRelabelingStrategy
    splitLambda::Float64             # The splitting parameter Lambda is used for regularization
    weightPenalty::Bool              # Whether to apply weight penalty
    overallBeta::Vector{Float64}     # Global regression coefficient
    llSplitCutoff::UInt              # Segmentation point threshold
    llSplitVariables::Vector{UInt}   # Variable index for segmentation
end

# 构造函数construct
function LLRegressionRelabelingStrategy(splitLambda::Float64, weightPenalty::Bool, overallBeta::Vector{Float64}, 
                                        llSplitCutoff::UInt, llSplitVariables::Vector{UInt})
    new(splitLambda, weightPenalty, overallBeta, llSplitCutoff, llSplitVariables)
end
using LinearAlgebra

# 定义一个重新标记函数
function relabel(strategy::LLRegressionRelabelingStrategy, samples::Vector{UInt}, data::Data, responsesBySample)
    numVariables = length(strategy.llSplitVariables)  # 获取分割变量的数量
    numDataPoints = length(samples)                   # 获取样本数量

    # 构建设计矩阵 X
    X = [ones(numDataPoints) [data.get(samples[i], strategy.llSplitVariables[j]) for i in 1:numDataPoints, j in 1:numVariables]]
    # 构建响应向量 Y
    Y = [data.getOutcome(samples[i]) for i in 1:numDataPoints]

    # 初始化叶子节点预测结果
    leafPredictions = zeros(numDataPoints)

    # 如果数据点少于分割阈值，使用全局回归系数
    if numDataPoints < strategy.llSplitCutoff
        eigenBeta = strategy.overallBeta
        leafPredictions = X * eigenBeta
    else
        # 否则，进行局部回归
        M = X' * X
        # 应用正则化
        if !strategy.weightPenalty
            normalization = tr(M) / (numVariables + 1)
            M += strategy.splitLambda * normalization * I
        else
            for j in 1:numVariables + 1
                M[j, j] += strategy.splitLambda * M[j, j]
            end
        end

        # 计算局部系数
        localCoefficients = M \ (X' * Y)
        # 根据局部系数计算预测值
        leafPredictions = X * localCoefficients
    end

    # 计算残差并更新响应
    for (i, sample) in enumerate(samples)
        predictionSample = leafPredictions[i]
        residual = predictionSample - data.getOutcome(sample)
        responsesBySample[sample, 1] = residual
    end

    return false
end
