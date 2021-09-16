using PyCall
using Conda
using Parameters


@with_kw mutable struct OptimizerCmaEsDeap
    opt::Any
    lambda_::Any
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
    update_count::Any
    ccov1::Any
    ccovmu::Any
    C::Any
    sigma::Any
    damps::Any
    diagD::Any
    B::Any
    BD::Any
    genomes::Any
end

function inititalize_optimizer(individual_size, configuration)
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size, configuration)

    optimizer = OptimizerCmaEsDeap(
        opt = opt,
        lambda_ = opt.strategy.lambda_,
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
        update_count = opt.strategy.update_count,
        ccov1 = opt.strategy.ccov1,
        ccovmu = opt.strategy.ccovmu,
        C = opt.strategy.C,
        sigma = opt.strategy.sigma,
        damps = opt.strategy.damps,
        diagD = opt.strategy.diagD,
        B = opt.strategy.B,
        BD = opt.strategy.BD,
        genomes = zeros(opt.strategy.lambda_, individual_size),
        
    )

    return optimizer, opt.strategy.eigenvectors, opt.strategy.indx
end

function ask(optimizer)

    genomes_list, population_size, individual_size, strategy = optimizer.opt.ask()

    # The genomes need to be reshaped into a MxN matrix.
    for i = 1:population_size
        for j = 1:individual_size
            optimizer.genomes[i, j] = (genomes_list[i])[j]
        end
    end

    return optimizer.genomes, strategy.randoms
end

function tell(optimizer, rewards)

    strategy = optimizer.opt.tell(rewards)

    optimizer.centroid = strategy.centroid
    optimizer.ps = strategy.ps
    optimizer.update_count = strategy.update_count
    optimizer.pc = strategy.pc
    optimizer.C = strategy.C
    optimizer.sigma = strategy.sigma
    optimizer.diagD = strategy.diagD
    optimizer.B = strategy.B
    optimizer.BD = strategy.BD

    return strategy.eigenvectors, strategy.indx
   
end
