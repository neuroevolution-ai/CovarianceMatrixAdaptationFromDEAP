using Test

include("optimizers/cma_es_deap.jl")

number_generations = 200
population_size = 10
sigma = 1.0
free_parameters = 20


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        genomes = ask(optimizer)

        rewards_training = rand(population_size)
        s = tell(optimizer, rewards_training)

        genomes_sorted = genomes[sortperm(rewards_training, rev=true),:]

        old_centroid = copy(s.centroid)
        centroid2 = genomes_sorted[1:s.mu,:]' * s.weights

        @test s.centroid ≈ centroid2 atol = 0.00001

        c_diff = centroid2 - old_centroid

        Q1 = s.Q1
        Q2 = s.Q2
        Q3 = s.Q3

        # @test ps ≈ ps2 atol = 0.00001

    end
end

println("Finished")
