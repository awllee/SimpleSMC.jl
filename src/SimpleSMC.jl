module SimpleSMC

using RNGPool
using Compat
import Compat.undef
import Compat.Statistics.mean
import Compat.Nothing
using Compat.Random

include("structures.jl")
include("subroutines.jl")
include("resample.jl")
include("algorithms.jl")
include("interface.jl")

export smc!, csmc!, SMCIO

end # module
