using PyCall
using Conda

function inititalize_optimizer(individual_size, configuration)
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size, configuration)
    return opt
end

function ask(optimizer)

    genomes_list, population_size, individual_size, strategy  = optimizer.ask()

    genomes = Matrix(undef, population_size, individual_size)

    # The genomes need to be reshaped into a MxN matrix.
    for i = 1:population_size
        for j = 1:individual_size
            genomes[i, j] = (genomes_list[i])[j]
        end
    end
    
    return genomes, strategy.B
end

function tell(optimizer, rewards)
    return optimizer.tell(rewards)
end
