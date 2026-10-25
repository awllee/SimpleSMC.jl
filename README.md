# SimpleSMC.jl

<!-- badges: start -->
[![CI](https://github.com/awllee/SimpleSMC.jl/workflows/CI/badge.svg)](https://github.com/awllee/SimpleSMC.jl/actions)
[![codecov](https://codecov.io/gh/awllee/SimpleSMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/awllee/SimpleSMC.jl)
<!-- badges: end -->

This package provides an implementation of a simple, single-threaded SMC algorithm.

It is not registered: its main purpose is to be extended in methodological research.

It implements a subset of the features of [SequentialMonteCarlo.jl](https://github.com/awllee/SequentialMonteCarlo.jl)


## Installation and compatibility

Requires Julia 1.10 or later and SequentialMonteCarlo 0.7. SimpleSMC remains
unregistered and can be installed from its Git repository. Until the companion
0.7 releases are registered, install all three from their default branches:

```julia
using Pkg
Pkg.add([
    PackageSpec(url="https://github.com/awllee/SequentialMonteCarlo.jl"),
    PackageSpec(url="https://github.com/awllee/SMCExamples.jl"),
    PackageSpec(url="https://github.com/awllee/SimpleSMC.jl"),
])
Pkg.test("SimpleSMC")
```

The algorithm remains single-threaded. RNGPool is replaced by Julia's task-local
`Random.default_rng()`. Model callbacks should accept `rng::AbstractRNG` (from
`Random`), or leave that argument untyped. Replace `setRNGs(seed)` with
`Random.seed!(seed)`. Seeded random streams differ from previous versions and
may change between Julia releases.
