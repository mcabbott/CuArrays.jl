using BinaryProvider
using CUDAnative

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
@assert isempty([a for a in ARGS if a != "--verbose"])

# online sources we can use
const bin_prefix = "https://github.com/JuliaGPU/CUDABuilder/releases/download/v0.1.4"
const resources = Dict(
    v"9.0" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDNN.v7.6.5-CUDA9.0-0.1.4.x86_64-linux-gnu.tar.gz", "7e55a9e0dc35295c11f0d069b1e852765b64d6ec2cb9ad2d9956269a46e87596"),
            Windows(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA9.0-0.1.4.x86_64-w64-mingw32.tar.gz", "3d5fd6eb56e64188254e9a7f5fd1ea725e8f83796e0fbc7664cedacf20d8aa87"),
        ),
    v"9.2" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDNN.v7.6.5-CUDA9.2-0.1.4.x86_64-linux-gnu.tar.gz", "f9f1efb9b89191647fe98470bcc2a2db77d4b5165cbe5627c0fa4894f59366a7"),
            Windows(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA9.2-0.1.4.x86_64-w64-mingw32.tar.gz", "fed63f16e2d4390112538ed28e7f5c7d64bcc66a81f024aaed1aed90e314b9bb"),
        ),
    v"10.0" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.0-0.1.4.x86_64-apple-darwin14.tar.gz", "702e02ea84a7092a9c9c96b76115647003660f6a7aaeaa604556d1d946287fdf"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.0-0.1.4.x86_64-linux-gnu.tar.gz", "49f016efc2c666163846e62d1a699a270a95e87ca4d15c37c41e14af644170c0"),
            Windows(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.0-0.1.4.x86_64-w64-mingw32.tar.gz", "c5b3691184004f08e0cd9a2ebc17623bdaaba2745ce0e359640f17ba207c3f38"),
        ),
    v"10.1" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.1-0.1.4.x86_64-apple-darwin14.tar.gz", "98289c4d4452023c743a492788dad5bb324c21535c9d9be49ceba180979c566d"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.1-0.1.4.x86_64-linux-gnu.tar.gz", "15bcd87b3e965014ca9a8b56ec5a101af08265377946dd1be4aa60bc3281a98a"),
            Windows(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.1-0.1.4.x86_64-w64-mingw32.tar.gz", "59217cc7f72e872a6aebf5fddd9b028bef50638e2677c57da2711f9aa58adc46"),
        ),
    v"10.2" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.2-0.1.4.x86_64-linux-gnu.tar.gz", "f0c37464796f160432b6c713981ea0dc9f6908dde2a8bf0814c990ef0a7f8ddc"),
            Windows(:x86_64) => ("$bin_prefix/CUDNN.v7.6.5-CUDA10.2-0.1.4.x86_64-w64-mingw32.tar.gz", "8757a6122eaef2c85aa0b0b6ec23f6275f9b99211899a800933373036976e184"),
        ),
)

# stuff we need to resolve
const cuarrays_prefix = Prefix(joinpath(@__DIR__, "usr"))
const cuarrays_products = if Sys.iswindows()
    width = Sys.WORD_SIZE
    [
        LibraryProduct(cuarrays_prefix, "cudnn$(width)_7", :libcudnn),
    ]
else
    [
        LibraryProduct(cuarrays_prefix, "libcudnn", :libcudnn),
    ]
end

# stuff we resolve in CUDAnative's prefix
const cudanative_prefix = Prefix(joinpath(dirname(dirname(pathof(CUDAnative))), "deps", "usr"))
const cudanative_products = if Sys.iswindows()
    # on Windows, library names are version dependent. That's a problem if were not using
    # BinaryBuilder, becuase that means we don't know the CUDA toolkit version yet!
    #
    # However, we can't just bail out here, because that would break users of packages
    # like Flux which depend on CuArrays but don't necessarily use it.
    try
        width = Sys.WORD_SIZE
        ver = CUDAnative.version()
        verstr = ver >= v"10.1" ? "$(ver.major)" : "$(ver.major)$(ver.minor)"
        [
            LibraryProduct(cudanative_prefix, "cufft$(width)_$(verstr)", :libcufft),
            LibraryProduct(cudanative_prefix, "curand$(width)_$(verstr)", :libcurand),
            LibraryProduct(cudanative_prefix, "cublas$(width)_$(verstr)", :libcublas),
            LibraryProduct(cudanative_prefix, "cusolver$(width)_$(verstr)", :libcusolver),
            LibraryProduct(cudanative_prefix, "cusparse$(width)_$(verstr)", :libcusparse),
        ]
    catch
        # just fail at runtime
        @error "On Windows, the CUDA toolkit version needs to be known at build time."
        @assert !CUDAnative.use_binarybuilder
        nothing
    end
else
    [
        LibraryProduct(cudanative_prefix, "libcufft", :libcufft),
        LibraryProduct(cudanative_prefix, "libcurand", :libcurand),
        LibraryProduct(cudanative_prefix, "libcublas", :libcublas),
        LibraryProduct(cudanative_prefix, "libcusolver", :libcusolver),
        LibraryProduct(cudanative_prefix, "libcusparse", :libcusparse),
    ]
end

const products = vcat(cuarrays_products, cudanative_products)
unsatisfied(products) = any(!satisfied(p; verbose=verbose) for p in products)

const depsfile = joinpath(@__DIR__, "deps.jl")

function main()
    rm(depsfile; force=true)

    use_binarybuilder = parse(Bool, get(ENV, "JULIA_CUDA_USE_BINARYBUILDER", "true"))
    if use_binarybuilder
        if try_binarybuilder()
            @assert !unsatisfied(products) && !unsatisfied(cudanative_products)
            return
        end
    end

    do_fallback()

    return
end

function try_binarybuilder()
    @info "Trying to provide CUDA libraries using BinaryBuilder"

    # get some libraries from CUDAnative
    if !CUDAnative.use_binarybuilder
        @warn "CUDAnative has not been built with BinaryBuilder, so CuArrays can't either."
        return false
    end
    @assert !unsatisfied(cudanative_products)

    # XXX: should it be possible to use CUDAnative without BB, but still download CUDNN?

    cuda_version = CUDAnative.version()
    @info "Working with CUDA $cuda_version"

    if !haskey(resources, cuda_version)
        @warn("Selected CUDA version is not available through BinaryBuilder.")
        return false
    end
    download_info = resources[cuda_version]

    # Install unsatisfied or updated dependencies:
    dl_info = choose_download(download_info, platform_key_abi())
    if dl_info === nothing && unsatisfied(cuarrays_products)
        # If we don't have a compatible .tar.gz to download, complain.
        # Alternatively, you could attempt to install from a separate provider,
        # build from source or something even more ambitious here.
        @warn("Your platform (\"$(Sys.MACHINE)\", parsed as \"$(triplet(platform_key_abi()))\") is not supported through BinaryBuilder.")
        return false
    end

    # If we have a download, and we are unsatisfied (or the version we're
    # trying to install is not itself installed) then load it up!
    if unsatisfied(cuarrays_products) || !isinstalled(dl_info...; prefix=cuarrays_prefix)
        # Download and install binaries
        install(dl_info...; prefix=cuarrays_prefix, force=true, verbose=verbose)
    end

    # Write out a deps.jl file that will contain mappings for our products
    write_deps_file(depsfile, products, verbose=verbose)

    open(depsfile, "a") do io
        println(io)
        println(io, "const use_binarybuilder = true")
    end

    return true
end

# assume that everything will be fine at run time
function do_fallback()
    @warn "Could not download CUDA dependencies; assuming they will be available at run time"

    open(depsfile, "w") do io
        println(io, "const use_binarybuilder = false")
        for p in products
            if p isa LibraryProduct
                # libraries are expected to be available on LD_LIBRARY_PATH
                println(io, "const $(variable_name(p)) = $(repr(first(p.libnames)))")
            end
        end
        println(io, """
            using Libdl
            function check_deps()
                Libdl.dlopen(libcufft)
                Libdl.dlopen(libcurand)
                Libdl.dlopen(libcublas)
                Libdl.dlopen(libcusolver)
                Libdl.dlopen(libcusparse)
                # CUDNN is an optional dependency
            end""")
    end

    return
end

main()
