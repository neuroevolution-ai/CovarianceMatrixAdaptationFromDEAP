using Test

include("optimizers/optimizer.jl")

number_generations = 100
population_size = 50
sigma = 1.0
free_parameters = 100


function main()
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        genomes, weights, mu = ask(optimizer)

        rewards_training = rand(population_size)
        centroid, ps, BD = tell(optimizer, rewards_training)

        p = sortperm(rewards_training)
        genomes_part = genomes[p][1:mu]

        centroid2 = weights * genomes_part

        @test 1 == 1

    end
end

main()

println("Finished")
