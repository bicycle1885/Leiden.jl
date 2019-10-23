"""
    modularity(graph::PartitionedGraph, γ::Float64)

Compute the modularity for a partitioned graph `graph` with resolution `γ`.

The resolution parameter is expected to be positive.

See M. E. J. Newman, 2006, <https://doi.org/10.1073%2Fpnas.0601602103>.
"""
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
