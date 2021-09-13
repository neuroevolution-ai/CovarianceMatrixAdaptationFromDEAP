using LinearAlgebra
using Parameters


@with_kw struct OptimizerCmaEs
    dim::Any
    chiN::Any
    mu::Any
    weights::Any
    mueff::Any
    cc::Any
    cs::Any
end

function tell(
    optimizer,
    rewards_training,
    genomes,
    B,
    diagD,
    sigma,
    ps,
    old_centroid,
    update_count,
    pc,
)

    genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

    centroid = genomes_sorted[1:optimizer.mu, :]' * optimizer.weights

    c_diff = centroid - old_centroid

    # Cumulation : update evolution path
    ps =
        (1 - optimizer.cs) .* ps +
        sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) ./ sigma *
        B *
        ((1 ./ diagD) .* B' * c_diff)

    hsig = float(
        norm(ps) / sqrt(1.0 - (1 - optimizer.cs)^(2 * (update_count + 1))) /
        optimizer.chiN < (1.4 + 2 / (optimizer.dim + 1)),
    )

    pc =
        (1 - optimizer.cc) * pc +
        hsig * sqrt(optimizer.cc * (2 - optimizer.cc) * optimizer.mueff) / sigma * c_diff

    return centroid, ps, pc

end
