using PyCall
using Test

@pyimport math
@pyimport numpy as np

math.sin(math.pi) - sin(pi)

n = 6
m = 4

A = rand(n, m)
println(A)

Apy = PyObject(A)
println(Apy)

for i in 1:size(A, 1)
    for j in 1:size(A, 2)
        @test A[i,j] == Apy[i,j]
    end
end

@test 1 == 1