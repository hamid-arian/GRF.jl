# Import necessary modules
import Pkg
Pkg.add("ConcurrentUtilities") 

using Pkg
Pkg.add("DataFrames")

include("data/Data.jl")
include("data/utility.jl")
include("forest/Forest.jl")
include("forest/ForestOptions.jl")
include("forest/ForestPredictor.jl")
include("forest/ForestPredictors.jl")
include("forest/ForestTrainer.jl")
include("forest/ForestTrainers.jl")
include("prediction/LocalLinearPredictionStrategy.jl")
include("relabelling/LLRegressionRelabelingStrategy.jl")
include("sampling/RandomSampler.jl")
include("sampling/SamplingOptions.jl")
include("splitting/MultiRegressionSplittingRule.jl")
include("splitting/RegressionSplittingRule.jl")
include("tree/Tree.jl")
include("tree/TreeOptions.jl")
include("tree/TreeTrainer.jl")



function main()
    # Load data
    # 从excel中导入数据
    using Pkg
   Pkg.add("XLSX")

   using XLSX
   storage,dataptr = XLSX.utility.loadData("C:\\Users\\86139\\Desktop", "Sheet1")  

   #创造data对象
   numRows=dataptr[0];
   numCols=dataptr[1]
   Data(storage, numRows, numCols)

   #分类,按照特征 ,#根据影响大小排序,samples一个随机整数，

   # 参数训练    根据给定的策略对特定的样本进行重新标记Used to control regression remarking
   struct LLRegressionRelabelingStrategy
    splitlambda::Float64#分裂参数
    weightpenalty::Bool#
    overallbeta::Vector{Int}#
    llsplitcutoff::Int#
    llsplitvariables#
   end
   relabeling_strategy = LLRegressionRelabelingStrategy(0.1,True,[0],10,storages)# 实例化结构体 parameter
    

   using DataFrames
   feature_columns = [
    :time, :highest_temperature, :lowest_temperature, :air_quality, :air_level
   ]

   database = DataFrame(storages, feature_columns)


  #每一列的只出现一次的数值的个数The number of values in each column that occur only once
  maxnumuniquevalues = maximum([nunique(col) for col in eachcol(database) if eltype(col) <: Number])

  # 分组规则
  struct RegressionSplittingRule
    maxnumuniquevalues::Int
    alpha::Float64
    imbalancepenalty::Float64
  end

    splitting_rule = RegressionSplittingRule(maxnumuniquevalues, 0.05, 0.01)# 实例化结构体，Instantiate the structure
    
    # Julia 中的索引从 1 开始，而不是从 0 开始，预测结果的策略
    #A local linear prediction strategy is defined and a series of indexes with specific features listed in the database are calculated.
    linearcorrectionindices = [findfirst(isequal(c), names(database)) for c in feature_columns]

    struct LocalLinearPredictionStrategy
        lambdas::Vector{Float64}
    end

    lambdas = 10 .^ range(-4, stop=2, length=10)
    predictionstrategy = LocalLinearPredictionStrategy(lambdas)# 实例化策略
    
   #构建决策树和森林，并训练创建 ForestTrainer 实例
    forest_trainer = ForestTrainer(relabeling_strategy, splitting_rule, predictionstrategy)
    ForestTrainer.train(forest_trainer,Data,getTrees(options))
  
    # Evaluate the model
    evaluation = evaluate_model(model, data)  

    # Print or return the evaluation results
    println("Model Evaluation: ", evaluation) 
    return evaluation
end

# Call the main function
main()
