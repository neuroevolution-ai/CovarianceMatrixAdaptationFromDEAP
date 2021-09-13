using LinearAlgebra


function tell(rewards_training, genomes, B, diagD, sigma, ps, old_centroid, update_count, pc, mu, weights, cs, mueff, chiN, dim, cc)

    genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

    centroid = genomes_sorted[1:mu, :]' * weights

    c_diff = centroid - old_centroid

    # Cumulation : update evolution path
    ps =
        (1 - cs) .* ps +
        sqrt(cs * (2 - cs) * mueff) ./ sigma * B * ((1 ./ diagD) .* B' * c_diff)

    hsig = float(
        norm(ps) / sqrt(1.0 - (1 - cs)^(2 * (update_count + 1))) / chiN <
        (1.4 + 2 / (dim + 1)),
    )

    pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) / sigma * c_diff

    return centroid, ps, pc

end
