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

# This method has a slightly tighter signature than the one in NNlib, all same eltype.
function NNlib.batched_mul!(C::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where {T<:CUBLAS.CublasFloat}
    if is_strided_cu(A) && is_strided_cu(B) && is_strided_cu(C)
        # Data is on GPU, and it's safe to call strides(A). gemm_strided_batched may be legal.
        batched_try_gemm!(C, A, B)

    elseif is_strided_cu(A) || is_strided_cu(B) || is_strided_cu(C)
        # This is hopeless, but best option is the fallback
        @debug "weird mix of CPU + GPU?"
        NNlib.batched_mul_generic!(C, A, B)

    else
        # All cases for CPU gemm! will come through here, is_strided_cu(A) compiles away:
        NNlib.batched_mul_cpu!(C, A, B)
    end
end

const batched_gemm_args = [
    (:(AbstractArray{T, 3}),       'N', :identity),
    (:(NNlib.BatchedTranspose{T}), 'T', :batched_transpose),
    (:(NNlib.BatchedAdjoint{T}),   'C', :batched_adjoint)
]

using NNlib: batched_mul!, BatchedTranspose, BatchedAdjoint, batched_transpose, batched_adjoint
using NNlib: _unbatch, _perm12

for (TA, transA, fA) in batched_gemm_args, (TB, transB, fB) in batched_gemm_args
    @eval function batched_try_gemm!(C::AbstractArray{T, 3}, A::$TA, B::$TB) where {T<:CUBLAS.CublasFloat}

        Abase, Bbase = _unbatch(A), _unbatch(B)

        # Best case, we can call batched_gemm! immediately:
        if Base.stride(Abase,1) == Base.stride(Bbase,1) == Base.stride(C,1) == 1
            CuArrays.CUBLAS.gemm_strided_batched!($transA, $transB, one(T), Abase, Bbase, zero(T), C)

        # Second-best, can we fix it by Perm.ing the base, and adjusing 'T' label?
        # But only if we won't produce BatchedTranspose(BatchedAdjoint(complex array)).
        elseif Base.stride(Abase,2) == 1 && !(T<:Complex && $TA<:BatchedAdjoint)
            newAbase = batched_transpose(_perm12(Abase))
            return batched_try_gemm!(C, $fA(newAbase), B)

        elseif Base.stride(Bbase,2) == 1 && !(T<:Complex && $TB<:BatchedAdjoint)
            newBbase = batched_transpose(_perm12(Bbase))
            return batched_try_gemm!(C, A, $fB(newBbase))

        # Fallback, e.g when Base.stride(A,3)==1
        else
            @debug "couldn't re-arrange strides for CUBLAS.gemm_strided_batched!" strides(A) strides(B) strides(C)
            NNlib.batched_mul_generic!(C, A, B)
        end
        C
    end
end


# This is obviously the wrong place for this! Not sure where it should go.
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


# Also the wong place for this, surely.
"""
    is_strided_cu(A)

This should return `true` for `A::CuArray`, and also for:
* Any `view(::CuArray)` or `reshape(::CuArray)` etc. which remains a `StridedArray`
* Any other wrapper for which `is_strided_cu(parent(A))`
* Except that `Adjoint(A)` is only unwrapped for real numbers.

Such wrappers include `PermutedDimsArray(::CuArray, ...)`,
but also those defined elsewhere (such as `NamedDimsArray`s)
which are assumed not to break strided-ness.

`Transpose` and `Adjoint` don't currently define `strides`, so for now they return `false`.
"""
is_strided_cu(A::CuArray) = true
is_strided_cu(A) = false
function is_strided_cu(A::AbstractArray)
    M = parentmodule(typeof(A))
    if parent(A) === A # Array, SparseMatrix, StaticArray
        false
    elseif M === Base || M === Core || M ===LinearAlgebra
        A isa StridedArray && is_strided_cu(parent(A))
    else
        is_strided_cu(parent(A)) # PermutedDimsArray, NamedDimsArray
    end
end

if hasmethod(Base.strides, Tuple{LinearAlgebra.Transpose})
    is_strided_cu(A::LinearAlgebra.Transpose) = is_strided(parent(A))
    is_strided_cu(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && is_strided(parent(A))
else
    is_strided_cu(A::LinearAlgebra.Transpose) = false
    is_strided_cu(A::LinearAlgebra.Adjoint) = false
end
