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
    neighbors,
    move_node!,
    drop_empty_communities!,
    reset_partition!,
    create_singleton_partition

include("karate.jl")

# The Leiden algorithm
# --------------------

function leiden(adjmat::AbstractMatrix{<:Real};
                resolution::Real = 1.0,
                randomness::Real = 0.01,
                partition::Partition = create_singleton_partition(size(adjmat, 1)))
    graph = PartitionedGraph(adjmat, partition = partition)
    return _leiden!(graph, Float64(resolution), Float64(randomness))
end

function _leiden!(graph::PartitionedGraph, γ::Float64, θ::Float64)
    stack = Partition[]
    @label loop
    move_nodes_fast!(graph, γ)
    @debug "nv = $(nv(graph)); nc = $(nc(graph)); H = $(H(graph, γ))"
    if nc(graph) != nv(graph)
        refined = refine_partition(graph, γ, θ)
        if nc(refined) == nv(refined)
            #@warn "not refined" nc(refined)
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
    return (quality = H(graph, γ), partition = flatten(stack))
end


# The Louvain algorithm
# ---------------------

function louvain(adjmat::AbstractMatrix{<:Real};
                 resolution::Real = 1.0,
                 partition::Partition = create_singleton_partition(size(adjmat, 1)))
    graph = PartitionedGraph(adjmat, partition = partition)
    return _louvain!(graph, Float64(resolution))
end

function _louvain!(graph::PartitionedGraph, γ::Float64)
    stack = Partition[]
    @label loop
    move_nodes!(graph, γ)
    @debug "nv = $(nv(graph)); nc = $(nc(graph)); H = $(H(graph, γ))"
    push!(stack, graph.partition)
    if nc(graph) != nv(graph)
        graph = aggregate_graph(graph)
        @goto loop
    end
    return (quality = H(graph, γ), partition = flatten(stack))
end

function move_nodes!(graph::PartitionedGraph, γ::Float64)
    H_old = H(graph, γ)
    nodes = collect(1:nv(graph))
    connected = Int[]
    total_weights = zeros(Float64, nc(graph))
    @label loop
    shuffle!(nodes)
    for u in nodes
        # compute total edge weights for each community
        empty!(connected)
        for v in neighbors(graph, u)
            i = graph.membership[v]
            if total_weights[i] == 0
                push!(connected, i)
            end
            total_weights[i] += graph.edge_weight[u,v]
        end

        # find the best community to which `u` belongs
        c_u = graph.cardinality[u]
        weight_u = graph.edge_weight[u,u]
        src = dst = graph.membership[u]
        weight_src = total_weights[src]
        size_src = graph.size[src]
        maxgain = 0.0
        for i in connected
            i == src && continue
            gain = total_weights[i] + weight_u - weight_src - γ * (graph.size[i] - size_src + c_u) * c_u
            if gain > maxgain
                dst = i
                maxgain = gain
            end
            total_weights[i] = 0
        end
        total_weights[src] = 0
        #@assert all(x == 0 for x in total_weights)

        if src != dst
            # move `u` to the best community and add its neighbors to the queue if needed
            move_node!(graph, u => dst)
        end
    end
    H_new = H(graph, γ)
    if H_new > H_old
        H_old = H_new
        @goto loop
    end
    return drop_empty_communities!(graph)
end

function move_nodes_fast!(graph::PartitionedGraph, γ::Float64)
    n = nv(graph)
    queue = shuffle(1:n)
    queued = BitSet(1:n)
    connected = Int[]
    total_weights = zeros(Float64, nc(graph))
    while !isempty(queue)
        u = popfirst!(queue)
        delete!(queued, u)

        # compute total edge weights for each community
        empty!(connected)
        for v in neighbors(graph, u)
            i = graph.membership[v]
            if total_weights[i] == 0
                push!(connected, i)
            end
            total_weights[i] += graph.edge_weight[u,v]
        end

        # find the best community to which `u` belongs
        c_u = graph.cardinality[u]
        weight_u = graph.edge_weight[u,u]
        src = dst = graph.membership[u]
        weight_src = total_weights[src]
        size_src = graph.size[src]
        maxgain = 0.0
        for i in connected
            i == src && continue
            gain = total_weights[i] + weight_u - weight_src - γ * (graph.size[i] - size_src + c_u) * c_u
            if gain > maxgain
                dst = i
                maxgain = gain
            end
            total_weights[i] = 0
        end
        total_weights[src] = 0
        #@assert all(x == 0 for x in total_weights)

        if src != dst
            # move `u` to the best community and add its neighbors to the queue if needed
            move_node!(graph, u => dst)
            for v in neighbors(graph, u)
                if graph.membership[v] != graph.membership[u] && v ∉ queued
                    push!(queue, v)
                    push!(queued, v)
                end
            end
        end
    end
    @assert isempty(queued)
    return drop_empty_communities!(graph)
end

function refine_partition(graph::PartitionedGraph, γ::Float64, θ::Float64)
    @assert γ > 0
    @assert θ > 0
    refined = PartitionedGraph(graph.edge_weight, cardinality = graph.cardinality)
    function is_well_connected(u::Int)
        i = graph.membership[u]
        c = graph.cardinality[u]
        threshold = γ * c * (graph.size[i] - c)
        x = 0.0
        for v in graph.partition[i]
            v == u && continue
            x += graph.edge_weight[u,v]
            if x ≥ threshold  # return as early as possible
                return true
            end
        end
        return false
    end
    function is_well_connected(u::Int, i::Int, between_weights::Vector{Float64})
        sz = refined.size[i]
        return between_weights[i] ≥ γ * sz * (graph.size[graph.membership[u]] - sz)
    end
    function is_singleton(u::Int)
        return length(refined.partition[refined.membership[u]]) == 1
    end
    total_weights = zeros(Float64, nc(refined))
    between_weights = zeros(Float64, nc(refined))
    for subset in graph.partition
        communities = Int[]
        logprobs = Float64[]
        indexes = Int[]
        for u in subset
            weight = 0.0
            for v in subset
                if v != u
                    weight += refined.edge_weight[u,v]
                end
            end
            between_weights[refined.membership[u]] = weight
        end
        for u in shuffle(subset)
            if !is_well_connected(u) || !is_singleton(u)
                continue
            end

            empty!(communities)
            for v in neighbors(refined, u)
                if v ∉ subset
                    continue
                end
                i = refined.membership[v]
                if total_weights[i] == 0
                    push!(communities, i)
                end
                total_weights[i] += refined.edge_weight[u,v]
            end

            c_u = refined.cardinality[u]
            weight_u = refined.edge_weight[u,u]
            src = refined.membership[u]
            weight_src = total_weights[src]
            size_src = refined.size[src]
            empty!(logprobs)
            empty!(indexes)
            for i in communities
                (i == src || !is_well_connected(u, i, between_weights)) && continue
                gain = total_weights[i] + weight_u - weight_src - γ * (refined.size[i] - size_src + c_u) * c_u
                if gain ≥ 0
                    push!(logprobs, 1/θ * gain)
                    push!(indexes, i)
                end
            end
            total_weights[communities] .= 0
            if isempty(indexes)
                continue
            end

            probs = exp.(logprobs .- logsumexp(logprobs))
            dst = indexes[sample(probs)]
            move_node!(refined, u => dst)

            for v in neighbors(refined, u)
                (v == u || v ∉ subset) && continue
                i = refined.membership[v]
                weight = refined.edge_weight[u,v]
                if i == dst
                    between_weights[src] -= weight
                    between_weights[dst] -= weight
                elseif i == src
                    between_weights[src] += weight
                    between_weights[dst] += weight
                else
                    between_weights[src] -= weight
                    between_weights[dst] += weight
                end
            end
        end
        for u in subset
            between_weights[refined.membership[u]] = 0
        end
        #@assert all(x == 0 for x in between_weights)
    end
    return drop_empty_communities!(refined)
end

function sample(probs::Vector{Float64})
    r = rand()
    p = 0.0
    i = 1
    while i < lastindex(probs)
        p += probs[i]
        if p > r
            return i
        end
        i += 1
    end
    return lastindex(probs)
end

function aggregate_graph(graph::PartitionedGraph)
    n = nc(graph)
    I = Int[]; J = Int[]; V = Float64[]
    cardinality = zeros(Int, n)
    connected = Int[]
    total_weights = zeros(Float64, n)
    for (i, community) in enumerate(graph.partition)
        cardinality[i] = graph.size[i]
        empty!(connected)
        for u in community, v in neighbors(graph, u)
            j = graph.membership[v]
            if i == j && u > v
                continue
            end
            if total_weights[j] == 0
                push!(connected, j)
            end
            total_weights[j] += graph.edge_weight[u,v]
        end
        for j in connected
            v = total_weights[j]
            push!(I, i)
            push!(J, j)
            push!(V, v)
            total_weights[j] = 0
        end
        #@assert all(x == 0 for x in total_weights)
    end
    return PartitionedGraph(sparse(I, J, V, n, n)::WeightMatrix, cardinality = cardinality)
end

function H(graph::PartitionedGraph, γ::Float64)
    quality = 0.0
    for (i, community) in enumerate(graph.partition)
        for u in community, v in community
            if u ≤ v
                quality += graph.edge_weight[u,v]
            end
        end
        n = graph.size[i]
        quality -= γ * div(n * (n - 1), 2)
    end
    return quality
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
