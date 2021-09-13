using Test

include("optimizers/cma_es_deap.jl")
include("optimizers/cma_es.jl")

number_generations = 100
population_size = 200
sigma = 1.5
free_parameters = 1000


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer_deap = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations

        # Ask optimizer for new population
        genomes, B, diagD, sigma, ps, old_centroid, update_count, pc = ask(optimizer_deap)

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizer new rewards
        strategy = tell(optimizer_deap, rewards_training)
        centroid, ps, pc = tell(rewards_training, genomes, B, diagD, sigma, ps, old_centroid, update_count, pc, strategy)

        @test strategy.centroid ≈ centroid atol = 0.00001
        @test strategy.ps ≈ ps atol = 0.00001
        @test strategy.pc ≈ pc atol = 0.00001
    end
end

println("Finished")
