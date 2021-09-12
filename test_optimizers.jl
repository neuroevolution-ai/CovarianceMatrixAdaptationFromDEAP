include("optimizers/optimizer.jl")

number_generations = 100
population_size = 50
sigma = 1.0
free_parameters = 100


function main()
    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)
    optimizer = inititalize_optimizer(free_parameters, optimizer_configuration)

    for generation = 1:number_generations
        ask(optimizer)

        rewards_training = rand(population_size)
        centroid, ps, BD = tell(optimizer, rewards_training)

        println(size(centroid))

    end
end

main()

println("Finished")
