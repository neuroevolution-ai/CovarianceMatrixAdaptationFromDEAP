using PyCall
using Conda
using Parameters


@with_kw mutable struct OptimizerCmaEsDeap
    dim::Any
    chiN::Any
    mu::Any
    weights::Any
    mueff::Any
    cc::Any
    cs::Any
    ps::Any
    pc::Any
    centroid::Any
    opt::Any
end

function inititalize_optimizer(individual_size, configuration)
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size, configuration)

    optimizer = OptimizerCmaEsDeap(
        dim = opt.strategy.dim,
        chiN = opt.strategy.chiN,
        mu = opt.strategy.mu,
        weights = opt.strategy.weights,
        mueff = opt.strategy.mueff,
        cc = opt.strategy.cc,
        cs = opt.strategy.cs,
        ps = opt.strategy.ps,
        pc = opt.strategy.pc,
        centroid = opt.strategy.centroid,
        opt = opt,
    )

    return optimizer
end

function ask(optimizer)

    genomes_list, population_size, individual_size, strategy = optimizer.opt.ask()

    genomes = Matrix(undef, population_size, individual_size)

    # The genomes need to be reshaped into a MxN matrix.
    for i = 1:population_size
        for j = 1:individual_size
            genomes[i, j] = (genomes_list[i])[j]
        end
    end

    return genomes, strategy.B, strategy.diagD, strategy.sigma, strategy.update_count
end

function tell(optimizer, rewards)

    strategy = optimizer.opt.tell(rewards)

    optimizer.centroid = strategy.centroid
    optimizer.ps = strategy.ps
    optimizer.pc = strategy.pc
end
