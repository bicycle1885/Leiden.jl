# Modularity
# ==========

struct Modularity
    resolution::Float64
end

function move_nodes!(graph::PartitionedGraph, model::Modularity)
    # immutable variables
    n = nv(graph)
    k = graph.node_weight
    A = graph.edge_weight
    η = 2 * model.resolution / sum(k)  # γ / m
    # mutable variables 
    σ = graph.membership
    s = graph.size
    nodes = collect(1:n)
    H_old = quality(graph, model)
    connected = Int[]
    connected_weights = zeros(nc(graph))
    @label loop
    shuffle!(nodes)
    for u in nodes
        empty!(connected)
        for v in neighbors(graph, u)
            i = σ[v]
            if iszero(connected_weights[i])
                push!(connected, i)
            end
            a = A[v,u]
            connected_weights[i] += u == v ? a : 2a
        end

        dst = src = σ[u]
        maxgain = connected_weights[dst] - η * k[u] * (s[dst] - k[u])
        for i in connected
            if i != src
                gain = connected_weights[i] - η * k[u] * s[i]
                if gain > maxgain
                    dst = i
                    maxgain = gain
                end
            end
            connected_weights[i] = 0
        end
        move_node!(graph, u => dst)
    end
    H_new = quality(graph, model)
    if H_new > H_old
        H_old = H_new
        @goto loop
    end
    return drop_empty_communities!(graph)
end

function move_nodes_fast!(graph::PartitionedGraph, model::Modularity)
    # immutable variables
    n = nv(graph)
    k = graph.node_weight
    A = graph.edge_weight
    η = 2 * model.resolution / sum(k)  # γ / m
    # mutable variables
    σ = graph.membership
    s = graph.size
    queue = shuffle(1:n)
    queued = BitSet(1:n)
    connected = Int[]
    connected_weights = zeros(nc(graph))
    while !isempty(queue)
        u = popfirst!(queue)
        delete!(queued, u)

        empty!(connected)
        for v in neighbors(graph, u)
            i = σ[v]
            if iszero(connected_weights[i])
                push!(connected, i)
            end
            a = A[v,u]
            connected_weights[i] += u == v ? a : 2a
        end

        dst = src = σ[u]
        maxgain = connected_weights[dst] - η * k[u] * (s[dst] - k[u])
        for i in connected
            if i != src
                gain = connected_weights[i] - η * k[u] * s[i]
                if gain > maxgain
                    dst = i
                    maxgain = gain
                end
            end
            connected_weights[i] = 0
        end
        move_node!(graph, u => dst)

        if dst != src
            for v in neighbors(graph, u)
                if σ[v] != σ[u] && v ∉ queued
                    push!(queue, v)
                    push!(queued, v)
                end
            end
        end
    end
    return drop_empty_communities!(graph)
end

function refine_partition(graph::PartitionedGraph, model::Modularity, θ::Float64)
    refined = PartitionedGraph(graph.node_weight, graph.edge_weight)
    P = refined.partition
    σ = refined.membership
    s = refined.size
    k = refined.node_weight
    A = refined.edge_weight
    γ = model.resolution
    η = 2 * γ / sum(k)
    internal_weights = zeros(nc(refined))
    connected = Int[]
    connected_weights = zeros(nc(refined))
    indexes = Int[]
    logprobs = Float64[]
    is_singleton(u) = length(P[σ[u]]) == 1
    for S in graph.partition
        function is_well_connected(C)  # NOTE: C must be a subset of S
            i = σ[first(C)]
            return internal_weights[i] ≥ η/4 * s[i] * (graph.size[graph.membership[S[1]]] - s[i])
        end
        # initialize internal weights of subset S
        for u in S
            w = 0.0
            for v in neighbors(refined, u)
                if v != u && v ∈ S
                    w += refined.edge_weight[v,u]
                end
            end
            internal_weights[σ[u]] = w
        end
        for u in shuffle(S)
            if !(is_singleton(u) && is_well_connected(u))
                continue
            end
            
            empty!(connected)
            for v in neighbors(graph, u)
                i = σ[v]
                if iszero(connected_weights[i])
                    push!(connected, i)
                end
                a = A[v,u]
                connected_weights[i] += u == v ? a : 2a
            end

            empty!(indexes)
            empty!(logprobs)
            src = σ[u]
            push!(indexes, src)
            push!(logprobs, 0.0)
            base = connected_weights[src] - η * k[u] * (s[src] - k[u])
            for i in connected
                if i != src && is_well_connected(P[i])
                    gain = connected_weights[i] - η * k[u] * s[i]
                    if gain ≥ base
                        push!(logprobs, 1/θ * (gain - base))
                        push!(indexes, i)
                    end
                end
                connected_weights[i] = 0
            end

            dst = indexes[logsample(logprobs)]
            move_node!(refined, u => dst)
            if dst != src
                internal_weights[src] = 0
                for v in neighbors(refined, u)
                    if v != u && v ∈ S && σ[v] == dst
                        internal_weights[dst] -= refined.edge_weight[v,u]
                    end
                end
            end
        end
        for u in S
            internal_weights[σ[u]] = 0
        end
    end
    return drop_empty_communities!(refined)
end

function logsample(logprobs)
    s = logsumexp(logprobs)
    r = rand()
    i = 1
    cumsum = 0.0
    while i < length(logprobs)
        cumsum += exp(logprobs[i] - s)
        if cumsum > r
            break
        end
        i += 1
    end
    return i
end

#=
function move_nodes!(graph::PartitionedGraph, model::Modularity)
    # constants (inside this function)
    n = nv(graph)
    A = graph.edge_weight
    σ = graph.membership
    k = degrees(graph)
    m = sum(k) / 2
    γ = model.resolution / m
    # variables
    H_old = quality(graph, model)
    nodes = collect(1:nv(graph))
    connected = Int[]
    connected_weights = zeros(nc(graph))
    community_degrees = zeros(nc(graph))
    for j in 1:n
        community_degrees[σ[j]] += k[j]
    end
    @label loop
    shuffle!(nodes)
    for u in nodes
        # compute total edge weights for connected communities
        empty!(connected)
        for v in neighbors(graph, u)
            i = σ[v]
            if connected_weights[i] == 0
                push!(connected, i)
            end
            a = A[v,u]
            connected_weights[i] += u == v ? a : 2a
        end

        # subtract the degree of u
        src = σ[u]
        community_degrees[src] -= k[u]

        # find the best community to which u belongs
        dst = src
        maxgain = connected_weights[dst] - γ * k[u] * community_degrees[dst]
        for i in connected
            gain = connected_weights[i] - γ * k[u] * community_degrees[i]
            if gain > maxgain
                dst = i
                maxgain = gain
            end
            connected_weights[i] = 0
        end

        # move u to the best community
        community_degrees[dst] += k[u]
        move_node!(graph, u => dst)
    end
    H_new = quality(graph, model)
    if H_new > H_old
        H_old = H_new
        @goto loop
    end
    return drop_empty_communities!(graph)
end

function move_nodes_fast!(graph::PartitionedGraph, model::Modularity)
    # constants (inside this function)
    n = nv(graph)
    A = graph.edge_weight
    σ = graph.membership
    k = degrees(graph)
    m = sum(k) / 2
    γ = model.resolution / m
    # variables
    queue = shuffle(1:n)
    queued = BitSet(1:n)
    connected = Int[]
    connected_weights = zeros(nc(graph))
    community_degrees = zeros(nc(graph))
    for j in 1:n
        community_degrees[σ[j]] += k[j]
    end
    while !isempty(queue)
        # pick the first node in the queue
        u = popfirst!(queue)
        delete!(queued, u)

        # compute total edge weights for connected communities
        empty!(connected)
        for v in neighbors(graph, u)
            i = σ[v]
            if connected_weights[i] == 0
                push!(connected, i)
            end
            a = A[v,u]
            connected_weights[i] += u == v ? a : 2a
        end

        # subtract the degree of u
        src = σ[u]
        community_degrees[src] -= k[u]

        # find the best community to which u belongs
        dst = src
        maxgain = connected_weights[dst] - γ * k[u] * community_degrees[dst]
        for i in connected
            gain = connected_weights[i] - γ * k[u] * community_degrees[i]
            if gain > maxgain
                dst = i
                maxgain = gain
            end
            connected_weights[i] = 0
        end

        # move u to the best community
        community_degrees[dst] += k[u]
        move_node!(graph, u => dst)

        # put u's neighbors into the queue
        if dst != src
            for v in neighbors(graph, u)
                if σ[v] != σ[u] && v ∉ queued
                    push!(queue, v)
                    push!(queued, v)
                end
            end
        end
    end
    return drop_empty_communities!(graph)
end
=#

function normalize_logprobs!(logprobs)
    x = logsumexp(logprobs)
    for i in 1:endof(logprobs)
        logprobs[i] = exp(logprobs[i] - x)
    end
    return logprobs
end

"""
    quality(graph::PartitionedGraph, model::Modularity)

Compute the modularity for a partitioned graph `graph`.

The resolution parameter is expected to be positive.

See M. E. J. Newman, 2006, <https://doi.org/10.1073%2Fpnas.0601602103>.
"""
function quality(graph::PartitionedGraph, model::Modularity)
    return modularity(graph, model.resolution)
end

function modularity(graph::PartitionedGraph, γ::Float64)
    A = graph.edge_weight
    σ = graph.membership
    # compute community statistics
    mm = 0.0  # mm == sum(A)
    d = zeros(nc(graph))  # total degrees
    w = zeros(nc(graph))  # total weights
    for j in 1:size(A, 2)
        for r in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[r]
            if i > j
                break  # skip redundant edges
            end
            a = A.nzval[r]
            if i == j
                mm += a
                d[σ[i]] += a
                w[σ[i]] += a
            else
                mm += 2a
                d[σ[i]] += a
                d[σ[j]] += a
                if σ[i] == σ[j]
                    w[σ[i]] += 2a
                end
            end
        end
    end
    # compute partition quality (i.e. modularity)
    quality = 0.0
    for c in 1:nc(graph)
        quality += w[c] - γ * d[c]^2 / mm
    end
    return quality / mm
end

# A naive version of modularity.
function modularity_naive(graph::PartitionedGraph, γ::Float64)
    A = graph.edge_weight
    σ = graph.membership
    m = sum(A) / 2        # total edge weight
    k = sum(A, dims = 1)  # degree
    # compute partition quality (i.e. modularity)
    quality = 0.0
    for i in 1:nv(graph), j in 1:nv(graph)
        if σ[i] == σ[j]
            quality += A[i,j] - γ * k[i] * k[j] / 2m
        end
    end
    return quality / 2m
end
