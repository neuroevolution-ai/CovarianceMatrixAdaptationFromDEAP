using LinearAlgebra


function tell(rewards_training, genomes, B, diagD, sigma, ps, old_centroid, update_count, pc, s)

    genomes_sorted = genomes[sortperm(rewards_training, rev = true), :]

    centroid = genomes_sorted[1:s.mu, :]' * s.weights

    c_diff = centroid - old_centroid

    # Cumulation : update evolution path
    ps =
        (1 - s.cs) .* ps +
        sqrt(s.cs * (2 - s.cs) * s.mueff) ./ sigma * B * ((1 ./ diagD) .* B' * c_diff)

    hsig = float(
        norm(ps) / sqrt(1.0 - (1 - s.cs)^(2 * (update_count + 1))) / s.chiN <
        (1.4 + 2 / (s.dim + 1)),
    )

    pc = (1 - s.cc) * pc + hsig * sqrt(s.cc * (2 - s.cc) * s.mueff) / sigma * c_diff

    return centroid, ps, pc

end
