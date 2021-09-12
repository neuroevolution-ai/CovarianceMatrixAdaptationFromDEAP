using Test

include("optimizers/cma_es_deap.jl")

number_generations = 200
population_size = 100
sigma = 1.0
free_parameters = 200


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        genomes, weights, mu = ask(optimizer)

        rewards_training = rand(population_size)
        centroid, ps, BD = tell(optimizer, rewards_training)

        genomes_sorted = genomes[sortperm(rewards_training, rev=true),:]

        centroid2 = genomes_sorted[1:mu,:]' * weights

        @test centroid ≈ centroid2 atol=0.00001

    end
end

println("Finished")
