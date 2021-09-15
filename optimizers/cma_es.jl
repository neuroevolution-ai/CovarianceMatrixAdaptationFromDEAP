using LinearAlgebra
using CUDA
using Parameters
using Distributions
using Random


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
    diagD::Any
    B::Any
    BD::Any
    genomes::Any
end

function ask(optimizer::OptimizerCmaEs, randoms)

    arz = rand(Normal(), size(optimizer.genomes))

    @test size(arz) == size(randoms)
    @test mean(arz) ≈ mean(randoms) atol = 0.1
    @test std(arz) ≈ std(randoms) atol = 0.01
    arz = copy(randoms)

    optimizer.genomes = optimizer.centroid' .+ (optimizer.sigma .* (arz * optimizer.BD'))

    return optimizer.genomes

end

function tell(optimizer::OptimizerCmaEs, rewards_training, eigenvectors1, indx1)

    genomes_sorted = optimizer.genomes[sortperm(rewards_training, rev = true), :]

    old_centroid = copy(optimizer.centroid)
    optimizer.centroid = genomes_sorted[1:optimizer.mu, :]' * optimizer.weights

    c_diff = optimizer.centroid - old_centroid

    # Cumulation : update evolution path
    optimizer.ps =
        (1 - optimizer.cs) .* optimizer.ps +
        sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) ./ optimizer.sigma *
        optimizer.B *
        ((1 ./ optimizer.diagD) .* optimizer.B' * c_diff)

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

    optimizer.sigma *=
        exp((norm(optimizer.ps) / optimizer.chiN - 1) * optimizer.cs / optimizer.damps)

    @test Hermitian(optimizer.C) ≈ optimizer.C atol = 0.00001

    optimizer.diagD, optimizer.B = eigen(Hermitian(optimizer.C))
    indx = sortperm(optimizer.diagD)

    # These lines are only to enable testing, since eigenvectors are not deterministic
    @test size(indx) == size(indx1)
    @test optimizer.diagD[indx] ≈ optimizer.diagD[indx1.+1] atol = 0.00001
    @test size(optimizer.B) == size(eigenvectors1)
    optimizer.B = copy(eigenvectors1)
    indx = copy(indx1 .+ 1)

    optimizer.diagD = optimizer.diagD[indx] .^ 0.5
    optimizer.B = optimizer.B[:, indx]
    optimizer.BD = optimizer.B .* optimizer.diagD'

end
