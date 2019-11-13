module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

import LinearAlgebra

using Adapt

using Libdl

using Requires


const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("CuArrays is not properly installed. Please run Pkg.build(\"CuArrays\")")
end

const libcutensor = Sys.iswindows() ? "cutensor" : "libcutensor"


## source code includes

include("memory.jl")
include("array.jl")
include("subarray.jl")
include("permuteddimsarray.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("matmul.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")

include("gpuarray_interface.jl")

include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

include("nnlib.jl")

include("deprecated.jl")


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

export has_cudnn, has_cutensor
has_cudnn() = Libdl.dlopen_e(libcudnn) !== C_NULL
has_cutensor() = Libdl.dlopen_e(libcutensor) !== C_NULL

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional() || !CUDAnative.functional()
        verbose && @warn "CuArrays.jl did not initialize because CUDAdrv.jl or CUDAnative.jl failed to"
        return
    end

    try
        # if we're not using BinaryBuilder, we can't be sure of everything at build-time
        if !use_binarybuilder
            silent || @warn """Automatic installation of the CUDA libraries failed; see $buildlog for more details
                               or call Pkg.build("CuArrays") to try again. Otherwise, you will need to install CUDA and make sure it is discoverable."""
        end

        check_deps()

        # library compatibility
        if has_cutensor()
            ver = CUTENSOR.version()
            if ver.major != 1
                silent || @warn("CuArrays.jl only supports CUTENSOR 1.x")
            end
        end

        # package integrations
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

        __init_memory__()

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CuArrays.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CuArrays.jl failed to initialize and will be unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end # module
