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

function generate_karate_club()
    # derived from http://konect.cc/networks/ucidata-zachary/
    weights = spzeros(34, 34)
    edges = [
        ( 1,  2), ( 1,  3), ( 2,  3), ( 1,  4), ( 2,  4),
        ( 3,  4), ( 1,  5), ( 1,  6), ( 1,  7), ( 5,  7),
        ( 6,  7), ( 1,  8), ( 2,  8), ( 3,  8), ( 4,  8),
        ( 1,  9), ( 3,  9), ( 3, 10), ( 1, 11), ( 5, 11),
        ( 6, 11), ( 1, 12), ( 1, 13), ( 4, 13), ( 1, 14),
        ( 2, 14), ( 3, 14), ( 4, 14), ( 6, 17), ( 7, 17),
        ( 1, 18), ( 2, 18), ( 1, 20), ( 2, 20), ( 1, 22),
        ( 2, 22), (24, 26), (25, 26), ( 3, 28), (24, 28),
        (25, 28), ( 3, 29), (24, 30), (27, 30), ( 2, 31),
        ( 9, 31), ( 1, 32), (25, 32), (26, 32), (29, 32),
        ( 3, 33), ( 9, 33), (15, 33), (16, 33), (19, 33),
        (21, 33), (23, 33), (24, 33), (30, 33), (31, 33),
        (32, 33), ( 9, 34), (10, 34), (14, 34), (15, 34),
        (16, 34), (19, 34), (20, 34), (21, 34), (23, 34),
        (24, 34), (27, 34), (28, 34), (29, 34), (30, 34),
        (31, 34), (32, 34), (33, 34),
    ]
    for (u, v) in edges
        weights[u,v] = weights[v,u] = 1
    end
    return weights
end

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
