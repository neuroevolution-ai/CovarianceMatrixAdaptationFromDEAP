using Test

include("optimizers/cma_es_deap.jl")
include("optimizers/cma_es.jl")
include("tools/compare_optimizer_states.jl")

number_generations = 100
population_size = 200
sigma = 1.5
free_parameters = 1000

tolerance = 0.00001


@testset "Optimizers" begin
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)

    # Initialize Optimizers
    # Optimizer1: CMA-ES optimizer of the Python deap package using PyCall
    # Optimizer2: Identical CMA-ES optimizer implemented in Julia
    optimizer1, eigenvectors1, indx1 = inititalize_optimizer(free_parameters, optimizer_configuration)
    optimizer2 = OptimizerCmaEs(free_parameters, optimizer_configuration, eigenvectors1, indx1)

    # Compare internal states of both optimizers
    compare_optimizer_states(optimizer1, optimizer2, tolerance)

    for generation = 1:number_generations

        # Ask optimizers for new population
        genomes1, randoms = ask(optimizer1)
        genomes2 = ask(optimizer2, randoms)

        @test genomes1 ≈ genomes2 atol = tolerance

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizers new rewards
        eigenvectors1, indx1 = tell(optimizer1, rewards_training)
        tell(optimizer2, rewards_training, eigenvectors1, indx1)

        # Compare internal states of both optimizers
        compare_optimizer_states(optimizer1, optimizer2, tolerance)

        # Test if C is a Hermitian matrix
        @test optimizer2.C ≈ Hermitian(optimizer2.C) atol = tolerance

    end
end

println("Finished")
