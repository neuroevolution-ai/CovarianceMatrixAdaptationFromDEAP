using LinearAlgebra
using Test

include("optimizers/cma_es_deap.jl")

number_generations = 100
population_size = 200
sigma = 1.5
free_parameters = 1000


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        genomes, B, diagD, sigma, ps, old_centroid, update_count = ask(optimizer)

        rewards_training = rand(population_size)
        s = tell(optimizer, rewards_training)

        genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

        centroid = genomes_sorted[1:s.mu, :]' * s.weights
        @test s.centroid ≈ centroid atol = 0.00001

        c_diff = centroid - old_centroid

        # Cumulation : update evolution path
        ps =
            (1 - s.cs) .* ps +
            sqrt(s.cs * (2 - s.cs) * s.mueff) ./ sigma * B * ((1 ./ diagD) .* B' * c_diff)
        @test s.ps ≈ ps atol = 0.00001

        hsig = float(
            norm(ps) / sqrt(1.0 - (1 - s.cs)^(2 * (update_count + 1))) / s.chiN <
            (1.4 + 2 / (s.dim + 1)),
        )
        @test s.hsig == hsig

    end
end

println("Finished")
