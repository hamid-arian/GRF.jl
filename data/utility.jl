using DelimitedFiles
using LinearAlgebra

# Split a sequence into a specified number of parts
function splitSequence(start::UInt, ends::UInt, numParts::UInt)
    result = UInt[]

    # Return range if only 1 part
    if numParts == 1
        return [start, ends + 1]
    end

    # Return vector from start to end+1 if more parts than elements
    if numParts > ends - start + 1
        return start:(ends + 1)
    end

    length = ends - start + 1
    partLengthShort = div(length, numParts)
    partLengthLong = ceil(Int, length / numParts)
    cutPos = length % numParts

    # Add long ranges
    for i in start:(start + cutPos * partLengthLong - 1)
        push!(result, i)
    end

    # Add short ranges
    for i in (start + cutPos * partLengthLong):(ends + 1)
        push!(result, i)
    end

    result
end

# Check if two doubles are approximately equal
equalDoubles(first::Float64, second::Float64, epsilon::Float64) = isnan(first) ? isnan(second) : abs(first - second) < epsilon

# Load data from a file
function loadData(fileName::String)
    # 使用 readdlm 函数读取文件内容。文件内容被假设为以空格分隔的浮点数，每行由换行符结束。
    data = readdlm(fileName, ' ', Float64, '\n')

    # 获取数据的行数和列数
    numRows, numCols = size(data)
    # Stores the data dimension as an array
    dim = [numRows, numCols]
    # Converts a two-dimensional array to a one-dimensional array and transposes
    storage = vec(data')

    # 返回处理后的一维数组和数据维度
    return storage, dim
end


# Set a specific value in the data
function setData(data::Tuple{Vector{Float64}, Vector{Int}}, row::Int, col::Int, value::Float64)
    storage, dim = data
    numRows = dim[1]

    storage[col * numRows + row] = value
    data = (storage, dim)
end
