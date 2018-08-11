module SimpleSMC

using RNGPool
import Statistics.mean
using Random

include("structures.jl")
include("subroutines.jl")
include("resample.jl")
include("algorithms.jl")
include("interface.jl")

export smc!, csmc!, SMCIO

end # module
