function move_nodes!(graph::PartitionedGraph, η::Float64)
    A = graph.edge_weight
    k = graph.node_weight
    a = graph.size
    σ = graph.membership
    n = nv(graph)
    H = quality(graph, η)
    nodes = collect(1:n)
    connected = Int[]
    connected_weights = zeros(nc(graph))
    @label loop
    shuffle!(nodes)
    for u in nodes
        empty!(connected)
        for v in neighbors(graph, u)
            if v == u
                # self-loops have no effects
                continue
            end
            i = σ[v]
            if iszero(connected_weights[i])
                push!(connected, i)
            end
            connected_weights[i] += A[v,u] * 2
        end

        dst = src = σ[u]
        maxgain = connected_weights[dst] - 2η * k[u] * (a[dst] - k[u])
        for i in connected
            if i != src
                gain = connected_weights[i] - 2η * k[u] * a[i]
                if gain > maxgain
                    dst = i
                    maxgain = gain
                end
            end
            connected_weights[i] = 0
        end
        move_node!(graph, u => dst)
    end
    H′ = quality(graph, η)
    if H′ > H
        H = H′
        @goto loop
    end
    drop_empty_communities!(graph)
end

function move_nodes_fast!(graph::PartitionedGraph, η::Float64)
    A = graph.edge_weight
    k = graph.node_weight
    a = graph.size
    σ = graph.membership
    n = nv(graph)
    H = quality(graph, η)
    connected = Int[]
    connected_weights = zeros(nc(graph))
    queue = shuffle(1:n)
    queued = BitSet(1:n)
    while !isempty(queue)
        u = popfirst!(queue)
        delete!(queued, u)

        empty!(connected)
        for v in neighbors(graph, u)
            if v == u
                # self-loops have no effects
                continue
            end
            i = σ[v]
            if iszero(connected_weights[i])
                push!(connected, i)
            end
            connected_weights[i] += A[u,v] * 2
        end

        dst = src = σ[u]
        maxgain = connected_weights[dst] - 2η * k[u] * (a[dst] - k[u])
        for i in connected
            if i != src
                gain = connected_weights[i] - 2η * k[u] * a[i]
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

function quality(graph::PartitionedGraph, η::Float64)
    # aggregate edge weights
    A = graph.edge_weight
    σ = graph.membership
    e = zeros(nc(graph))
    for j in 1:nv(graph)
        for r in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[r]
            if i > j
                break
            end
            w = A.nzval[r]
            if i == j
                e[σ[i]] += w
            elseif σ[i] == σ[j]
                e[σ[i]] += 2w
            end
        end
    end
    # tally community scores
    a = graph.size
    #@show e a
    quality = 0.0
    for c in 1:nc(graph)
        quality += e[c] - η * a[c]^2
    end
    return quality
end

function refine_partition(graph::PartitionedGraph, η::Float64, θ::Float64)
    k = graph.node_weight
    A = graph.edge_weight
    refined = PartitionedGraph(k, A)
    P = refined.partition
    σ = refined.membership
    a = refined.size
    internal_weights = zeros(nc(refined))
    connected = Int[]
    connected_weights = zeros(nc(refined))
    indexes = Int[]
    logprobs = Float64[]
    is_singleton(u) = length(P[σ[u]]) == 1
    for S in graph.partition
        function is_well_connected(C)
            i = σ[first(C)]
            return internal_weights[i] ≥ η * a[i] * (graph.size[graph.membership[S[1]]] - a[i])
        end
        # initialize internal weights of S
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
                if v == u
                    # self-loops have no effects
                    continue
                end
                i = σ[v]
                if iszero(connected_weights[i])
                    push!(connected, i)
                end
                connected_weights[i] += A[u,v]
            end

            empty!(indexes)
            empty!(logprobs)
            src = σ[u]
            push!(indexes, src)
            push!(logprobs, 0.0)
            base = connected_weights[src] - 2η * k[u] * (a[src] - k[u])
            for i in connected
                if i != src && is_well_connected(P[i])
                    gain = connected_weights[i] - 2η * k[u] * a[i]
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
    drop_empty_communities!(refined)
    return refined
end

function logsample(logprobs::Vector{Float64})
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
