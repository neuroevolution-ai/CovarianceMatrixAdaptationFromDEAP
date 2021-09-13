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
    optimizer1 = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations

        # Ask optimizer for new population
        genomes1, B1, diagD1, sigma1, ps1, old_centroid1, update_count1, pc1 = ask(optimizer1)

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizer new rewards
        centroid1, ps1_, pc1_, mu1, weights1, cs1, mueff1, chiN1, dim1, cc1 = tell(optimizer1, rewards_training)

        # Initialize OptimizerCmaEs        
        optimizer2 = OptimizerCmaEs(dim=dim1, chiN=chiN1, mu=mu1, weights=weights1, mueff=mueff1, cc=cc1, cs=cs1)

        centroid2, ps2, pc2 = tell(optimizer2, rewards_training, genomes1, B1, diagD1, sigma1, ps1, old_centroid1, update_count1, pc1)

        @test centroid1 ≈ centroid2 atol = 0.00001
        @test ps1_ ≈ ps2 atol = 0.00001
        @test pc1_ ≈ pc2 atol = 0.00001
    end
end

println("Finished")
