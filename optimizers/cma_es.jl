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
    update_count::Any
    ccov1::Any
    ccovmu::Any
    C::Any
    sigma::Any
    damps::Any
end

function tell(optimizer::OptimizerCmaEs, rewards_training, genomes, B, diagD)

    genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

    old_centroid = copy(optimizer.centroid)
    optimizer.centroid = genomes_sorted[1:optimizer.mu, :]' * optimizer.weights

    c_diff = optimizer.centroid - old_centroid

    # Cumulation : update evolution path
    optimizer.ps =
        (1 - optimizer.cs) .* optimizer.ps +
        sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) ./ optimizer.sigma *
        B *
        ((1 ./ diagD) .* B' * c_diff)

    hsig = float(
        norm(optimizer.ps) /
        sqrt(1.0 - (1 - optimizer.cs)^(2 * (optimizer.update_count + 1))) /
        optimizer.chiN < (1.4 + 2 / (optimizer.dim + 1)),
    )

    optimizer.update_count += 1

    optimizer.pc =
        (1 - optimizer.cc) * optimizer.pc +
        hsig * sqrt(optimizer.cc * (2 - optimizer.cc) * optimizer.mueff) / optimizer.sigma *
        c_diff

    # Update covariance matrix
    artmp = genomes_sorted[1:optimizer.mu, :]' .- old_centroid

    optimizer.C =
        (
            1 - optimizer.ccov1 - optimizer.ccovmu +
            (1 - hsig) * optimizer.ccov1 * optimizer.cc * (2 - optimizer.cc)
        ) * optimizer.C +
        optimizer.ccov1 * optimizer.pc * optimizer.pc' +
        optimizer.ccovmu * (optimizer.weights' .* artmp) * artmp' / optimizer.sigma^2

    optimizer.sigma *= exp((norm(optimizer.ps) / optimizer.chiN - 1) * optimizer.cs / optimizer.damps)

end
