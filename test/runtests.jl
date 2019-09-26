using Leiden
using Test
using Random: Random
using SparseArrays: spzeros
using StatsBase: summarystats

include("reference.jl")

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

generate_karate_club() = Leiden.karate_matrix()

# Relative error.
relerror(x, y) = abs(x - y) / max(abs(x), abs(y))

@testset "Leiden.jl" begin
    Random.seed!(1234)
    γ = 0.25
    result = Leiden.leiden(generate_two_community(), resolution = γ)
    @test result.quality == 6 * (1 - γ)
    @test result.partition == [[1,2,3], [4,5,6]]

    Random.seed!(1234)
    γ = 0.25
    result = Leiden.louvain(generate_two_community(), resolution = γ)
    @test result.quality == 6 * (1 - γ)
    @test result.partition == [[1,2,3], [4,5,6]]

    Random.seed!(1234)
    γ = 0.05
    karate = generate_karate_club()
    stats = summarystats([Leiden.leiden(karate, resolution = γ).quality for _ in 1:1000])
    stats_ref = summarystats([Reference.leiden(karate, resolution = γ).quality for _ in 1:1000])
    #@show relerror(stats.mean, stats_ref.mean) relerror(stats.max, stats_ref.max)
    #@show relerror(stats.min, stats_ref.min) relerror(stats.median, stats_ref.median)
    @test relerror(stats.mean,   stats_ref.mean)   < 1e-3
    @test relerror(stats.max,    stats_ref.max)    < 1e-3
    @test relerror(stats.min,    stats_ref.min)    < 1e-3
    @test relerror(stats.median, stats_ref.median) < 1e-3

    Random.seed!(1234)
    γ = 0.05
    karate = generate_karate_club()
    stats = summarystats([Leiden.louvain(karate, resolution = γ).quality for _ in 1:1000])
    stats_ref = summarystats([Reference.louvain(karate, resolution = γ).quality for _ in 1:1000])
    #@show relerror(stats.mean, stats_ref.mean) relerror(stats.max, stats_ref.max)
    #@show relerror(stats.min, stats_ref.min) relerror(stats.median, stats_ref.median)
    @test relerror(stats.mean,   stats_ref.mean)   < 1e-3
    @test relerror(stats.max,    stats_ref.max)    < 1e-3
    #@test relerror(stats.min,    stats_ref.min)    < 1e-3
    @test relerror(stats.median, stats_ref.median) < 1e-3
end
