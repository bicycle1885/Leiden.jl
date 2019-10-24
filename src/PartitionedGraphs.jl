# Partitioned Graphs
# ==================

module PartitionedGraphs

using SparseArrays
using LinearAlgebra

# Nodes and communities are identified by a unique integer of Int, and edges
# are weighted by a floating-point number of Float64.
const Partition = Vector{Vector{Int}}
const WeightVector = Vector{Float64}
const WeightMatrix = SparseMatrixCSC{Float64,Int}

struct PartitionedGraph
    # graph (immutable)
    node_weight::WeightVector
    edge_weight::WeightMatrix

    # partition (mutable)
    partition::Partition
    size::WeightVector
    membership::Vector{Int}
end

function PartitionedGraph(
        node_weight::AbstractVector{<:Real},
        edge_weight::AbstractMatrix{<:Real};
        partition::Partition = create_singleton_partition(length(node_weight)),
        check::Bool = true,)
    n = Base.size(node_weight, 1)
    if check
        check_adjacent_matrix(edge_weight)
        check_nodeweights(n, node_weight)
        check_partition(n, partition)
    end
    size = zeros(length(partition))
    membership = zeros(Int, n)
    for (i, community) in enumerate(partition)
        size[i] = weighted_sum(community, node_weight)
        membership[community] .= i
    end
    return PartitionedGraph(node_weight, edge_weight, partition, size, membership)
end

create_singleton_partition(n::Integer) = [[i] for i in 1:n]

function check_adjacent_matrix(adjmat::AbstractMatrix{<:Real})
    m, n = size(adjmat)
    if m != n
        throw(ArgumentError("invalid adjacent matrix: not a square matrix"))
    end
    if n == 0
        throw(ArgumentError("invalid adjacent matrix: empty matrix"))
    end
    if !issymmetric(adjmat)
        throw(ArgumentError("invalid adjacent matrix: not an symmetric matrix"))
    end
    if adjmat isa SparseMatrixCSC
        if any(v < 0 for v in adjmat.nzval)
            throw(ArgumentError("invalid adjacent matrix: found negative weight(s)"))
        end
    else
        if any(v < 0 for v in adjmat)
            throw(ArgumentError("invalid adjacent matrix: found negative weight(s)"))
        end
    end
    return nothing
end

function check_nodeweights(n::Integer, weights::AbstractVector{<:Real})
    if n != length(weights)
        throw(ArgumentError("invalid node weight: mismatching length"))
    end
    if any(x < 0 for x in weights)
        throw(ArgumentError("invalid node weight: found negative value(s)"))
    end
    return nothing
end

function check_partition(n::Integer, partition::Partition)
    found = BitSet()
    for community in partition, u in community
        if u ∈ found
            throw(ArgumentError("invalid partition: found duplicated node"))
        end
        if !(1 ≤ u ≤ n)
            throw(ArgumentError("invalid partition: found out-of-bounds node"))
        end
        push!(found, u)
    end
    if length(found) != n
        throw(ArgumentError("invalid partition: found missing node"))
    end
    return nothing
end

# number of vertices
nv(graph::PartitionedGraph) = length(graph.node_weight)

# number of communities
nc(graph::PartitionedGraph) = length(graph.partition)

# weighted degrees
function degrees(graph::PartitionedGraph)
    A = graph.edge_weight
    n = nv(graph)
    k = zeros(n)
    for j in 1:n, r in A.colptr[j]:A.colptr[j+1]-1
        k[j] += A.nzval[r]
    end
    return k
end

# neighbors of node `u`
function neighbors(graph::PartitionedGraph, u::Int)
    A = graph.edge_weight
    return @view A.rowval[A.colptr[u]:A.colptr[u+1]-1]
end

# communities connected with `u`
function connected_communities(graph::PartitionedGraph, u::Int)
    A = graph.total_weight
    return @view A.rowval[A.colptr[u]:A.colptr[u+1]-1]
end

# Move node `u` to `dst`.
function move_node!(graph::PartitionedGraph, (u, dst)::Pair{Int,Int})
    @assert 1 ≤ dst ≤ nc(graph) + 1
    src = graph.membership[u]
    if src == dst
        # no movement
        return graph
    end
    community_src = graph.partition[src]
    pos = findfirst(isequal(u), community_src)::Int
    if dst > nc(graph)
        community_dst = Int[]
        push!(graph.partition, community_dst)
        push!(graph.size, 0)
    else
        community_dst = graph.partition[dst]
    end
    deleteat!(community_src, pos)
    push!(community_dst, u)
    w = graph.node_weight[u]
    graph.size[src] -= w
    graph.size[dst] += w
    graph.membership[u] = dst
    return graph
end

# Drop empty communities from the graph.
function drop_empty_communities!(graph::PartitionedGraph)
    empty = Int[]
    for (i, community) in enumerate(graph.partition)
        if isempty(community)
            push!(empty, i)
        end
    end
    deleteat!(graph.partition, empty)
    deleteat!(graph.size, empty)
    for (i, community) in enumerate(graph.partition)
        graph.membership[community] .= i
    end
    return graph
end

# Normalize partition.
function normalize!(graph::PartitionedGraph)
     # TODO
end

function reset_partition!(graph::PartitionedGraph, partition::Partition)
    check_partition(nv(graph), partition)
    empty!(graph.partition)
    empty!(graph.size)
    for (i, community) in enumerate(partition)
        push!(graph.partition, copy(community))
        push!(graph.size, weighted_sum(community, graph.node_weight))
        graph.membership[community] .= i
    end
    return graph
end

function weighted_sum(subset::Vector{Int}, weights::WeightVector)
    return isempty(subset) ? 0.0 : sum(weights[i] for i in subset)
end

end
