using BenchmarkTools
using LinearAlgebra
using CUDA
using Test

free_parameters = 1000

#C = rand(ComplexF32, free_parameters, free_parameters)
C = rand(free_parameters, free_parameters)
C += C'
C_GPU = CuArray(convert(Matrix{Float32}, C))

@btime val, vec = eigen(C)
@btime CUDA.@sync val_GPU, vec_GPU = CUDA.CUSOLVER.syevd!('V', 'U', C_GPU)

#val, vec = eigen(C)
#val_GPU, vec_GPU = CUDA.CUSOLVER.syevd!('V', 'U', C_GPU)

#@test val ≈ Array(val_GPU)

println("Finished")