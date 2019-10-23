module Leiden

using Random:
    shuffle,
    shuffle!
using SparseArrays:
    sparse,
    spzeros
using StatsFuns:
    logsumexp

include("PartitionedGraphs.jl")
using .PartitionedGraphs:
    PartitionedGraph,
    Partition,
    WeightMatrix,
    nv, nc,
    degrees,
    neighbors,
    move_node!,
    drop_empty_communities!,
    reset_partition!,
    create_singleton_partition

include("modularity.jl")
include("cpm.jl")
include("optimizer.jl")
include("karate.jl")

# The Leiden algorithm
# --------------------

function leiden(adjmat::AbstractMatrix{<:Real};
                resolution::Real = 1.0,
                model::Symbol = :modularity,
                randomness::Real = 0.01,
                partition::Partition = create_singleton_partition(size(adjmat, 1)))
    γ = Float64(resolution)
    θ = Float64(randomness)
    if model === :modularity
        # weighted by degree
        weight = vec(sum(Float64, adjmat, dims = 1))
        m2 = sum(weight)
        η = γ / m2
        scale = inv(m2)
    elseif model == :cpm
        # unit weight
        weight = ones(size(adjmat, 2))
        η = γ
        scale = 1.0
    else
        throw(ArgumentError("unsupported model: '$(model)'"))
    end
    graph = PartitionedGraph(weight, adjmat, partition = partition)
    stack = Partition[]
    @label loop
    move_nodes_fast!(graph, η)
    if nc(graph) != nv(graph)
        refined = refine_partition(graph, η, θ)
        if nc(refined) == nv(refined)
            @goto finish
        end
        push!(stack, refined.partition)
        graph′ = aggregate_graph(refined)
        partition = [Int[] for _ in 1:nc(graph)]
        for (i, community) in enumerate(refined.partition)
            u = first(community)
            j = graph.membership[u]
            push!(partition[j], i)
        end
        reset_partition!(graph′, partition)
        graph = graph′
        @goto loop
    end
    @label finish
    push!(stack, graph.partition)
    return (quality = quality(graph, η) * scale, partition = flatten(stack))
end


# The Louvain algorithm
# ---------------------

function louvain(adjmat::AbstractMatrix{<:Real};
                 resolution::Real = 1.0,
                 model::Symbol = :modularity,
                 partition::Partition = create_singleton_partition(size(adjmat, 1)))
    γ = Float64(resolution)
    if model === :modularity
        # weighted by degree
        weight = vec(sum(Float64, adjmat, dims = 1))
        m2 = sum(weight)
        η = γ / m2
        scale = inv(m2)
    elseif model === :cpm
        # unit weight
        weight = ones(size(adjmat, 2))
        η = γ
        scale = 1.0
    else
        throw(ArgumentError("unsupported model: '$(model)'"))
    end
    graph = PartitionedGraph(weight, adjmat, partition = partition)
    stack = Partition[]
    @label loop
    move_nodes!(graph, η)
    push!(stack, graph.partition)
    if nc(graph) != nv(graph)
        graph = aggregate_graph(graph)
        @goto loop
    end
    flattened = PartitionedGraph(weight, adjmat, partition = flatten(stack))
    return (quality = quality(graph, η) * scale, partition = flatten(stack))
end

function aggregate_graph(graph::PartitionedGraph)
    n = nc(graph)
    I = Int[]; J = Int[]; V = Float64[]
    node_weight = zeros(n)
    connected = Int[]
    connected_weights = zeros(Float64, n)
    for (i, community) in enumerate(graph.partition)
        node_weight[i] = graph.size[i]
        empty!(connected)
        for u in community, v in neighbors(graph, u)
            j = graph.membership[v]
            if i == j && u > v
                #continue
            end
            if iszero(connected_weights[j])
                push!(connected, j)
            end
            connected_weights[j] += graph.edge_weight[u,v]
        end
        for j in connected
            v = connected_weights[j]
            push!(I, i)
            push!(J, j)
            push!(V, v)
            connected_weights[j] = 0
        end
        #@assert all(x == 0 for x in connected_weights)
    end
    return PartitionedGraph(node_weight, sparse(I, J, V, n, n))
end

function flatten(stack::Vector{Partition})
    k = lastindex(stack)
    result = copy(stack[k])
    k -= 1
    while k ≥ 1
        partition = stack[k]
        for (i, indices) in enumerate(result)
            result[i] = mapfoldl(i -> partition[i], vcat, indices, init = Int[])
        end
        k -= 1
    end
    sort!(result, by = length)
    foreach(sort!, result)
    return result
end

end # module
