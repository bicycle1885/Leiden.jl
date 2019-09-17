using Leiden
using Test
using Random: Random
using SparseArrays: spzeros

function generate_two_community()
    weights = spzeros(6, 6)
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
        weights[u,v] = weights[v,u] = 1
    end
    return weights
end

@testset "Leiden.jl" begin
    Random.seed!(1234)

    γ = 0.25
    adjmat = generate_two_community()
    result = Leiden.leiden(adjmat, resolution = γ)
    @test result.quality == 6 * (1 - γ)
    @test result.partition == [[1,2,3], [4,5,6]]
end
