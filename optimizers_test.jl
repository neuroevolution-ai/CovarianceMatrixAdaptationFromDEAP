using Test

include("optimizers/cma_es_deap.jl")
include("optimizers/cma_es.jl")

number_generations = 100
population_size = 200
sigma = 1.5
free_parameters = 1000


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)

    # Initialize Optimizers
    # Optimizer1: CMA-ES optimizer of the Python deap package using PyCall
    # Optimizer2: Identical CMA-ES optimizer implemented in Julia
    optimizer1 = inititalize_optimizer(free_parameters, optimizer_configuration)
    optimizer2 = OptimizerCmaEs(
        dim = optimizer1.dim,
        chiN = optimizer1.chiN,
        mu = optimizer1.mu,
        weights = optimizer1.weights,
        mueff = optimizer1.mueff,
        cc = optimizer1.cc,
        cs = optimizer1.cs,
        ps = optimizer1.ps,
        pc = optimizer1.pc,
        centroid = optimizer1.centroid,
    )

    for generation = 1:number_generations

        # Ask optimizer for new population
        genomes1, B1, diagD1, sigma1, update_count1 = ask(optimizer1)

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizers new rewards
        tell(optimizer1, rewards_training)
        tell(optimizer2, rewards_training, genomes1, B1, diagD1, sigma1, update_count1)

        # Compare internal states of both optimizers
        @test optimizer1.centroid ≈ optimizer2.centroid atol = 0.00001
        @test optimizer1.ps ≈ optimizer2.ps atol = 0.00001
        @test optimizer1.pc ≈ optimizer2.pc atol = 0.00001
    end
end

println("Finished")
