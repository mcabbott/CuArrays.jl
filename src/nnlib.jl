using NNlib

# Activation functions

@cufunc σ(x) = ifelse(x < -80, zero(x), one(x) / (one(x) + exp(-x)))

@cufunc function logσ(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end

@cufunc elu(x, α = one(x)) =
  ifelse(x ≥ 0, x/1, α * (exp(x) - one(x)))

@cufunc swish(x) = x * σ(x)

@cufunc function gelu(x)
  λ = oftype(x/1, √(2/π))
  α = oftype(x/1, 0.044715)
  h = oftype(x/1, 0.5)
  h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

@cufunc function selu(x)
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
end

@cufunc softplus(x) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))


# Batched matrix multiplication

const batched_gemm_args = [
    (:(AbstractArray{T, 3}), 'N', :identity),
    (:(NNlib.BatchedTranspose{T, <:AbstractArray{T, 3}}), 'T', :batched_transpose),
    (:(NNlib.BatchedAdjoint{T, <:AbstractArray{T, 3}}), 'C', :batched_adjoint)
]

using NNlib: batched_mul!, BatchedTranspose, BatchedAdjoint, batched_transpose, batched_adjoint

for (TA, transA, fA) in batched_gemm_args, (TB, transB, fB) in batched_gemm_args
    @eval function NNlib.batched_mul!(C::CuArray{T, 3}, A::$TA, B::$TB) where {T<:CUBLAS.CublasFloat}

        Abase, Bbase = NNlib._unbatch(A), NNlib._unbatch(B)

        # Best case, we can call batched_gemm! immediately:
        if Base.stride(Abase,1) == Base.stride(Bbase,1) == Base.stride(C,1) == 1
            CuArrays.CUBLAS.gemm_strided_batched!($transA, $transB, one(T), NNlib._unbatch(A), NNlib._unbatch(B), zero(T), C)

        # Second-best, can we fix it by Perm.ing the base, and adjusing 'T' label?
        # But only if we won't produce BatchedTranspose(BatchedAdjoint(complex array)).
        elseif Base.stride(Abase,2) == 1 && !(T<:Complex && $TA<:NNlib.BatchedAdjoint)
            newAbase = NNlib.batched_transpose(PermutedDimsArray(Abase, (2,1,3)))
            return NNlib.batched_mul!(C, $fA(newAbase), B)
        elseif Base.stride(Bbase,2) == 1 && !(T<:Complex && $TB<:NNlib.BatchedAdjoint)
            newBbase = NNlib.batched_transpose(PermutedDimsArray(Bbase, (2,1,3)))
            return NNlib.batched_mul!(C, A, $fB(newBbase))

        # Fallback, e.g when Base.stride(A,3)==1
        else
            NNlib.batched_mul_generic!(C, A, B)
        end
        C
    end
end

# This is obviously the wrong place for this! Not sure where it should go.
# Base.unsafe_convert(::Type{CUDAdrv.CuPtr{T}}, A::PermutedDimsArray) where {T} = 
#     Base.unsafe_convert(CUDAdrv.CuPtr{T}, parent(A))
# Recursive version, will handle e.g. NamedDimsArray
function Base.unsafe_convert(::Type{CUDAdrv.CuPtr{T}}, A::AbstractArray) where {T}
    if A === parent(A)
        throw(MethodError(Base.unsafe_convert, Tuple{CUDAdrv.CuPtr{T}, typeof(A)}))
    else
        return Base.unsafe_convert(CUDAdrv.CuPtr{T}, parent(A))
    end
end


# This is https://github.com/JuliaLang/julia/pull/35304, here just for testing now:
Base.similar(A::PermutedDimsArray, T::Type, dims::Base.Dims) = similar(parent(A), T, dims)
