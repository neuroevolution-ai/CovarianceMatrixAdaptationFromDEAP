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
        copy(optimizer1.mu),
        copy(optimizer1.weights),
        copy(optimizer1.mueff),
        copy(optimizer1.cc),
        copy(optimizer1.cs),
        copy(optimizer1.centroid),
        copy(optimizer1.update_count),
        copy(optimizer1.ccov1),
        copy(optimizer1.ccovmu),
        copy(optimizer1.sigma),
        copy(optimizer1.damps),
        copy(optimizer1.diagD),
        copy(optimizer1.B),
        copy(optimizer1.BD),
        copy(optimizer1.genomes),
    )

    @test optimizer1.dim ≈ optimizer2.dim atol = 0.00001
    @test optimizer1.pc ≈ optimizer2.pc atol = 0.00001
    @test optimizer1.ps ≈ optimizer2.ps atol = 0.00001
    @test optimizer1.chiN ≈ optimizer2.chiN atol = 0.00001
    @test optimizer1.C ≈ optimizer2.C atol = 0.00001

    for generation = 1:number_generations

        # Ask optimizers for new population
        genomes1, randoms = ask(optimizer1)
        genomes2 = ask(optimizer2, randoms)

        @test genomes1 ≈ genomes2 atol = 0.00001

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizers new rewards
        eigenvectors1, indx1 = tell(optimizer1, rewards_training)
        tell(optimizer2, rewards_training, eigenvectors1, indx1)

        # Compare internal states of both optimizers
        @test optimizer1.centroid ≈ optimizer2.centroid atol = 0.00001
        @test optimizer1.ps ≈ optimizer2.ps atol = 0.00001
        @test optimizer1.update_count ≈ optimizer2.update_count atol = 0.00001
        @test optimizer1.pc ≈ optimizer2.pc atol = 0.00001
        @test optimizer1.C ≈ optimizer2.C atol = 0.00001
        @test optimizer1.sigma ≈ optimizer2.sigma atol = 0.00001
        @test optimizer1.diagD ≈ optimizer2.diagD atol = 0.00001
        @test optimizer1.B ≈ optimizer2.B atol = 0.00001
        @test optimizer1.BD ≈ optimizer2.BD atol = 0.00001

        # Test if C is a Hermitian matrix
        @test optimizer2.C ≈ Hermitian(optimizer2.C) atol = 0.00001

    end
end

println("Finished")
