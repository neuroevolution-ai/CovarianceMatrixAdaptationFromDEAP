using LinearAlgebra
using Parameters


@with_kw mutable struct OptimizerCmaEs
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
end

function tell(optimizer, rewards_training, genomes, B, diagD, sigma, update_count)

    genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

    old_centroid = copy(optimizer.centroid)
    optimizer.centroid = genomes_sorted[1:optimizer.mu, :]' * optimizer.weights

    c_diff = optimizer.centroid - old_centroid

    # Cumulation : update evolution path
    optimizer.ps =
        (1 - optimizer.cs) .* optimizer.ps +
        sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) ./ sigma *
        B *
        ((1 ./ diagD) .* B' * c_diff)

    hsig = float(
        norm(optimizer.ps) / sqrt(1.0 - (1 - optimizer.cs)^(2 * (update_count + 1))) /
        optimizer.chiN < (1.4 + 2 / (optimizer.dim + 1)),
    )

    optimizer.pc =
        (1 - optimizer.cc) * optimizer.pc +
        hsig * sqrt(optimizer.cc * (2 - optimizer.cc) * optimizer.mueff) / sigma * c_diff

end
