include("optimizers/optimizer.jl")

number_generations = 100
population_size = 50
sigma = 1.0
N = 100


function main()
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    free_parameters = 100

    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        genomes = ask(optimizer)

        rewards_training = rand(population_size)
        tell(optimizer, rewards_training)
    end
end

main()

println("Finished")
