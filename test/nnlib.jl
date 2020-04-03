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

using NNlib: is_strided, are_strided, storage_type
using LinearAlgebra
@testset "NNlib storage_type etc." begin

    M = cu(ones(10,10))

    @test is_strided(M)
    @test is_strided(view(M, 1:2:5,:))
    @test is_strided(PermutedDimsArray(M, (2,1)))

    @test !is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !is_strided((M.+im)')
    @test !is_strided(Diagonal(cu(ones(3))))

    @test storage_type(M) == CuArray{Float32,2,Nothing}
    @test storage_type(reshape(view(M, 1:2:10,:), 10,:)) == CuArray{Float32,2,Nothing}

end

end
