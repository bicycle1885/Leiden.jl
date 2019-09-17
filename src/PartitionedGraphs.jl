module PartitionedGraphs

using SparseArrays
using LinearAlgebra

const Partition = Vector{Vector{Int}}
const WeightMatrix = SparseMatrixCSC{Float64,Int}

struct PartitionedGraph
    # graph (immutable)
    edge_weight::WeightMatrix
    cardinality::Vector{Int}

    # partition (mutable)
    partition::Partition
    size::Vector{Int}
    membership::Vector{Int}
end

function PartitionedGraph(
        adjmat::AbstractMatrix{<:Real};
        cardinality::Vector{Int} = create_unit_cardinality(nv(adjmat)),
        partition::Partition = create_singleton_partition(nv(adjmat)),)
    check_adjacent_matrix(adjmat)
    n = nv(adjmat)
    check_cardinality(n, cardinality)
    check_partition(n, partition)
    size = zeros(Int, length(partition))
    membership = zeros(Int, n)
    for (i, community) in enumerate(partition)
        size[i] = weighted_sum(community, cardinality)
        membership[community] .= i
    end
    return PartitionedGraph(adjmat, cardinality, partition, size, membership)
end

# number of vertices
nv(A::AbstractMatrix) = size(A, 1)
nv(graph::PartitionedGraph) = nv(graph.edge_weight)

# number of communities
nc(graph::PartitionedGraph) = length(graph.partition)

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

# generators of the default parameter
create_unit_cardinality(n::Integer) = fill(1, n)
create_singleton_partition(n::Integer) = [[i] for i in 1:n]

function check_adjacent_matrix(adjmat::AbstractMatrix)
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

function check_cardinality(n::Integer, cardinality::Vector{Int})
    if n != length(cardinality)
        throw(ArgumentError("invalid cardinality: mismatching length"))
    end
    if any(x ≤ 0 for x in cardinality)
        throw(ArgumentError("invalid cardinality: found non-positive value"))
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

# Move node `u` to `dst`.
function move_node!(graph::PartitionedGraph, (u, dst)::Pair{Int,Int})
    @assert 1 ≤ dst ≤ nc(graph) + 1
    src = graph.membership[u]
    if src == dst
        # no movement
        return graph
    end
    cardinality = graph.cardinality[u]
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
    graph.size[src] -= cardinality
    graph.size[dst] += cardinality
    graph.membership[u] = dst
    return graph
end

# Drop empty communities from the graph.
function drop_empty_communities!(graph::PartitionedGraph)
    empty = Int[]
    for (i, community) in enumerate(graph.partition)
        @assert isempty(community) == (graph.size[i] == 0)
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

function reset_partition!(graph::PartitionedGraph, partition::Partition)
    check_partition(nv(graph), partition)
    empty!(graph.partition)
    empty!(graph.size)
    for (i, community) in enumerate(partition)
        push!(graph.partition, copy(community))
        push!(graph.size, weighted_sum(community, graph.cardinality))
        graph.membership[community] .= i
    end
    return graph
end

function weighted_sum(xs::Vector{Int}, weights::Vector{Int})
    return isempty(xs) ? 0 : sum(weights[x] for x in xs)
end

end
