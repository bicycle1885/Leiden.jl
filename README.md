# Leiden

[![Build Status](https://travis-ci.com/bicycle1885/Leiden.jl.svg?branch=master)](https://travis-ci.com/bicycle1885/Leiden.jl)
[![Codecov](https://codecov.io/gh/bicycle1885/Leiden.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bicycle1885/Leiden.jl)

A Julia implementation of the Leiden algorithm for community detection.

Traag, Vincent A., Ludo Waltman, and Nees Jan van Eck. "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific reports* 9 (2019).
<https://doi.org/10.1038/s41598-019-41695-z>

## Usage

```julia
# Create an adjacent matrix.
using SparseArrays
A = spzeros(6, 6)
edges = [
    (1, 2),
    (2, 3),
    (3, 1),
    (4, 5),
    (5, 6),
    (6, 4),
    (1, 4),
]
for (u, v) in edges
    A[u,v] = A[v,u] = 1
end

# Run the Leiden algorithm.
using Random
using Leiden
Random.seed!(1234)
result = Leiden.leiden(A, resolution = 0.25)
@show result
# result = (quality = 4.5, partition = Array{Int64,1}[[1, 2, 3], [4, 5, 6]])
```
