# Reference implementations for testing
# =====================================

module Reference

using LightGraphs: nv, vertices, neighbors, has_edge, add_edge!
using SimpleWeightedGraphs: SimpleWeightedGraph
using StatsFuns: logsumexp
using Random: shuffle

# Types used throughout the reference implementations
#   vertex identifier: Int
#   edge weight: Float64
const Graph = SimpleWeightedGraph{Int,Float64}
const Partition = Vector{BitSet}

# The Louvain algorithm.
function louvain(graph::Graph;
                 partition::Partition = create_singleton_partition(graph),
                 resolution::Float64 = 1.0)
    @assert resolution > 0
    check_partition(graph, partition)
    stack = Partition[]
    cardinality = fill(1, nv(graph))
    @label loop
    partition = move_nodes(graph, cardinality, partition, resolution)
    check_partition(graph, partition)
    push!(stack, partition)
    if length(partition) != nv(graph)
        H_base = H(graph, cardinality, partition, resolution)
        graph, cardinality = aggregate_graph(graph, cardinality, partition)
        partition = create_singleton_partition(graph)
        @assert H(graph, cardinality, partition, resolution) ≈ H_base
        check_partition(graph, partition)
        @goto loop
    end
    return (quality = H(graph, cardinality, partition, resolution),
            partition = flatten(stack),)
end

function move_nodes(graph::Graph, cardinality::Vector{Int},
                    partition::Partition, resolution::Float64)
    γ = resolution
    partition = map(copy, partition)
    @label loop
    H_init = H(graph, cardinality, partition, γ)
    for u in shuffle(vertices(graph))
        dst, gain = find_best_community(graph, cardinality, partition, u, γ)
        if gain > 0
            move_node!(partition, u => dst)
        end
    end
    if H(graph, cardinality, partition, γ) > H_init
        @goto loop
    end
    return partition
end

function leiden(adjmat::AbstractMatrix;
                partition::Partition = create_singleton_partition(adjmat),
                resolution::Float64 = 1.0)
    return leiden(Graph(adjmat), partition = partition, resolution = resolution)
end

function leiden(graph::Graph;
                partition::Partition = create_singleton_partition(graph),
                resolution::Float64 = 1.0)
    @assert resolution > 0
    check_partition(graph, partition)
    stack = Partition[]
    cardinality = fill(1, nv(graph))
    @label loop
    partition = move_nodes_fast(graph, cardinality, partition, resolution)
    check_partition(graph, partition)
    if length(partition) != nv(graph)
        partition′ = refine_partition(graph, cardinality, partition, resolution)
        push!(stack, partition′)
        check_partition(graph, partition′)
        H_base = H(graph, cardinality, partition′, resolution)
        graph, cardinality = aggregate_graph(graph, cardinality, partition′)
        @assert H(graph, cardinality, create_singleton_partition(graph), resolution) ≈ H_base
        partition = map(partition) do comm
            new = create_empty_community()
            for (u, comm′) in zip(vertices(graph), partition′)
                if issubset(comm′, comm)
                    push!(new, u)
                end
            end
            @assert !isempty(new)
            new
        end
        check_partition(graph, partition)
        @goto loop
    end
    push!(stack, partition)
    return (quality = H(graph, cardinality, partition, resolution),
            partition = flatten(stack),)
end

function move_nodes_fast(graph::Graph, cardinality::Vector{Int},
                         partition::Partition, resolution::Float64)
    γ = resolution
    partition = map(copy, partition)
    queue = shuffle(1:nv(graph))
    while !isempty(queue)
        u = popfirst!(queue)
        dst, gain = find_best_community(graph, cardinality, partition, u, γ)
        if gain > 0
            move_node!(partition, u => dst)
            dst = findfirst(comm -> u ∈ comm, partition)  # move_node! may move communities
            ns = [v for v in neighbors(graph, u) if v ∉ partition[dst]]
            append!(queue, setdiff(ns, queue))
        end
    end
    return partition
end

function refine_partition(graph::Graph, cardinality::Vector{Int},
                          partition::Partition, resolution::Float64)
    partition′ = create_singleton_partition(graph)
    for comm in partition
        partition′ = merge_nodes_subset(graph, cardinality, partition′, comm, resolution)
    end
    return partition′
end

function merge_nodes_subset(graph::Graph, cardinality::Vector{Int},
                            partition::Partition, subset::BitSet, γ::Float64)
    # well-connectedness
    function is_well_connected(u::Int)
        a = tally_weights(graph, create_singleton_community(u), setdiff(subset, u))
        b = cardinality[u]
        c = tally_cardinality(subset, cardinality)
        return a ≥ γ * b * (c - b)
    end
    function is_well_connected(comm::BitSet)
        a = tally_weights(graph, comm, setdiff(subset, comm))
        b = tally_cardinality(comm, cardinality)
        c = tally_cardinality(subset, cardinality)
        return a ≥ γ * b * (c - b)
    end
    θ = 0.01
    partition = map(copy, partition)
    for u in subset
        if !is_well_connected(u)
            continue
        end
        i = findfirst(comm -> u ∈ comm, partition)
        @assert !isnothing(i)
        if is_singleton(partition[i])
            logprobs = fill(-Inf, length(partition))
            for (i, comm) in enumerate(partition)
                if issubset(comm, subset) && is_well_connected(comm)
                    gain = ΔH(graph, cardinality, partition, u => i, γ)
                    logprobs[i] = gain ≥ 0 ? 1/θ * gain : -Inf
                end
            end
            if !isfinite(logsumexp(logprobs))
                break
            end
            move_node!(partition, u => sample(logprobs))
        end
    end
    return partition
end

function sample(logprobs::Vector{Float64})
    probs = exp.(logprobs .- logsumexp(logprobs))
    r = rand()
    i = 1
    cumprob = 0.0
    while i < lastindex(probs)
        cumprob += probs[i]
        if cumprob ≥ r
            break
        end
        i += 1
    end
    return i
end

# Find the best community to which `u` is assigned.
function find_best_community(graph::Graph, cardinality::Vector{Int},
                             partition::Partition, u::Int, γ::Float64)
    # new community?
    dst = lastindex(partition) + 1
    push!(partition, create_empty_community())
    maxgain = ΔH(graph, cardinality, partition, u => dst, γ)
    pop!(partition)
    # or existing communities?
    for i in 1:lastindex(partition)
        val = ΔH(graph, cardinality, partition, u => i, γ)
        if val > maxgain
            dst = i
            maxgain = val
        end
    end
    return dst, maxgain
end

function ΔH(graph::Graph, cardinality::Vector{Int},
            partition::Partition, move::Pair{Int,Int}, γ::Float64)
    u, dst = move
    partition′ = map(enumerate(partition)) do (i, comm)
        if i == dst
            # `u` can be in `comm`
            comm = copy(comm)
            push!(comm, u)
        elseif u ∈ comm
            comm = copy(comm)
            delete!(comm, u)
        end
        return comm
    end
    return H(graph, cardinality, partition′, γ) - H(graph, cardinality, partition, γ)
end

function H(graph::Graph, cardinality::Vector{Int}, partition::Partition, γ::Float64)
    result = 0.0
    for (i, comm) in enumerate(partition)
        weight = tally_weights(graph, partition, i, i)
        n = isempty(comm) ? 0 : sum(cardinality[u] for u in comm)
        result += weight - γ * div(n * (n - 1), 2)
    end
    return result
end

function move_node!(partition::Partition, move::Pair{Int,Int})
    u, dst = move
    src = findfirst(comm -> u ∈ comm, partition)
    @assert !isnothing(src)
    if src == dst
        return partition
    end
    if dst > lastindex(partition)
        # new community
        push!(partition, create_singleton_community(u))
    else
        # existing community
        push!(partition[dst], u)
    end
    delete!(partition[src], u)
    if isempty(partition[src])
        deleteat!(partition, src)
    end
    return partition
end

function aggregate_graph(graph::Graph, cardinality::Vector{Int}, partition::Partition)
    @assert all(!isempty, partition)
    nv = length(partition)
    aggregated = SimpleWeightedGraph(nv)
    for i in 1:nv, j in i:nv
        weight = tally_weights(graph, partition, i, j)
        @assert weight ≥ 0
        if weight > 0
            add_edge!(aggregated, i, j, weight)
        end
    end
    return aggregated, [sum(cardinality[u] for u in comm) for comm in partition]
end

function tally_weights(graph::Graph)
    n = nv(graph)
    weight = 0.0
    for i in 1:n, j in i:n
        weight += graph.weights[i,j]
    end
    return weight
end

function tally_weights(graph::Graph, partition::Partition, i::Int, j::Int)
    weight = 0.0
    if i == j
        comm = partition[i]
        for u in comm, v in comm
            if u ≤ v
                weight += graph.weights[u,v]
            end
        end
    else
        for u in partition[i], v in partition[j]
            weight += graph.weights[u,v]
        end
    end
    return weight
end

function tally_weights(graph::Graph, comm1::BitSet, comm2::BitSet)
    @assert isdisjoin(comm1, comm2)
    weight = 0.0
    for u in comm1, v in comm2
        if has_edge(graph, u, v)
            weight += graph.weights[u,v]
        end
    end
    return weight
end

function isdisjoin(set1::BitSet, set2::BitSet)
    for x in set1, y in set2
        if x == y
            return false
        end
    end
    return true
end

function tally_cardinality(comm::BitSet, cardinality::Vector{Int})
    return isempty(comm) ? 0 : sum(cardinality[u] for u in comm)
end

function flatten(partitions::Vector{Partition})
    result = map(copy, last(partitions))
    for i in lastindex(partitions):-1:2
        partition = partitions[i-1]
        for j in 1:lastindex(result)
            result[j] = mapfoldl(x -> partition[x], union, result[j],
                                 init = create_empty_community())
        end
    end
    sort!(result, by = length)
    return result
end

function create_singleton_partition(graph::Graph)
    return [create_singleton_community(u) for u in vertices(graph)]
end

function create_singleton_partition(adjmat::AbstractMatrix)
    return [create_singleton_community(u) for u in 1:size(adjmat, 1)]
end

function create_empty_community()
    return BitSet()
end

function create_singleton_community(u::Int)
    return BitSet(u)
end

function is_singleton(comm::BitSet)
    return length(comm) == 1
end

function check_partition(graph::Graph, partition::Partition)
    n = nv(graph)
    found = BitSet()
    for comm in partition
        if !isempty(comm ∩ found)
            throw(AssertionError("non-disjoint partition: $(collect(comm ∩ found))"))
        elseif maximum(comm) > n
            throw(AssertionError("out of bounds partition: $([u for u in comm if u > n])"))
        end
        union!(found, comm)
    end
    if length(found) != n
        throw(AssertionError("non-exhaustive partition: $(collect(setdiff(BitSet(1:n), found)))"))
    end
end

end
