@testset "NNlib" begin

@testset "batched_mul" begin
    using NNlib: batched_mul, batched_adjoint, batched_transpose

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test cu(C) ≈ batched_mul(cu(A), cu(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test cu(Ct) ≈ batched_mul(batched_transpose(cu(A)), cu(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test cu(Ca) ≈ batched_mul(cu(A), batched_adjoint(cu(B)))
end

using CuArrays: is_strided_cu
using LinearAlgebra
@testset "is_strided_cu" begin

    M = cu(ones(10,10))

    @test is_strided_cu(M)
    @test is_strided_cu(view(M, 1:2:5,:))
    @test is_strided_cu(PermutedDimsArray(M, (2,1)))

    @test !is_strided_cu(reshape(view(M, 1:2:10,:), 10,:))
    @test !is_strided_cu((M.+im)')
    @test !is_strided_cu(ones(10,10))
    @test !is_strided_cu(Diagonal(ones(3)))

    #=
    using NamedDims
    @test is_strided(NamedDimsArray(M,(:a, :b))) # and 0.029 ns, 0 allocations
    =#

end

end
