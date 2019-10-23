"""
    cpm(model::CPM, graph::PartitionedGraph)

Compute the partition quality of the constant Potts model (CPM) for a
partitioned graph `graph` with resolution `γ`.

The resolution parameter is expected to be between 0 and 1.

See V. A. Traag and P. Van Dooren, 2011, <https://doi.org/10.1103/PhysRevE.84.016114>.
"""
function cpm(graph::PartitionedGraph, γ::Float64)
    A = graph.edge_weight
    k = graph.node_weight
    σ = graph.membership
    # compute community statistics
    a = zeros(nc(graph))
    e = zeros(nc(graph))
    for j in 1:size(A, 2)
        a[σ[j]] += k[j]
        for r in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[r]
            if i > j
                break  # skip redundant edges
            end
            w = A.nzval[r]
            if i == j
                e[σ[i]] += w
            elseif σ[i] == σ[j]
                e[σ[i]] += 2w
            end
        end
    end
    #@show e a
    # compute partition quality (i.e. CPM)
    quality = 0.0
    for c in 1:nc(graph)
        quality += e[c] - γ * a[c]^2
    end
    return quality
end

# A naive version of cpm.
function cpm_naive(graph::PartitionedGraph, γ::Float64)
    A = graph.edge_weight
    σ = graph.membership
    quality = 0.0
    for i in 1:nv(graph), j in 1:nv(graph)
        if σ[i] == σ[j]
            quality += A[i,j] - γ
        end
    end
    return quality
end
