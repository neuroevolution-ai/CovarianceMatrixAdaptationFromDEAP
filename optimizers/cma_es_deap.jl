using PyCall
using Conda

function inititalize_optimizer(individual_size, configuration)
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size, configuration)

    return opt,
    opt.strategy.dim,
    opt.strategy.chiN,
    opt.strategy.mu,
    opt.strategy.weights,
    opt.strategy.mueff,
    opt.strategy.cc,
    opt.strategy.cs,
    opt.strategy.ps,
    opt.strategy.pc
end

function ask(optimizer)

    genomes_list, population_size, individual_size, strategy = optimizer.ask()

    genomes = Matrix(undef, population_size, individual_size)

    # The genomes need to be reshaped into a MxN matrix.
    for i = 1:population_size
        for j = 1:individual_size
            genomes[i, j] = (genomes_list[i])[j]
        end
    end

    return genomes,
    strategy.B,
    strategy.diagD,
    strategy.sigma,
    strategy.centroid,
    strategy.update_count
end

function tell(optimizer, rewards)

    strategy = optimizer.tell(rewards)

    return strategy.centroid,
    strategy.ps,
    strategy.pc,
    strategy.mu,
    strategy.weights,
    strategy.cs,
    strategy.mueff,
    strategy.chiN,
    strategy.dim,
    strategy.cc
end
