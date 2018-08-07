import Compat.hasmethod
import SequentialMonteCarlo.SMCModel

# the fields of this struct are used by the SMC algorithm
# they are not intended for use by users of the package
mutable struct _SMCInternal{Particle, ParticleScratch}
  zetaAncs::Vector{Particle}
  as::Vector{Int64}
  lws::Vector{Float64}
  scratch1::Vector{Float64}
  scratch2::Vector{Float64}
  vecOnes::Vector{Float64}
  particleScratch::ParticleScratch
end

## constructor for _SMCInternal
function _SMCInternal{Particle, ParticleScratch}(N::Int64, n::Int64) where
  {Particle, ParticleScratch}

  lws = Vector{Float64}(undef, N)
  as = Vector{Int64}(undef, N)
  fill!(as, 1)
  scratch1 = Vector{Float64}(undef, N)
  scratch2 = Vector{Float64}(undef, N)
  vecOnes = ones(N)
  zetaAncs = Vector{Particle}(undef, N)
  for i in 1:N
    zetaAncs[i] = Particle()
  end

  ## assign user-defined particle scratch space
  @assert ParticleScratch == Nothing || !isbits(ParticleScratch)
  particleScratch = ParticleScratch()

  return _SMCInternal(zetaAncs, as, lws, scratch1, scratch2, vecOnes,
    particleScratch)
end

## SMC input / output struct. Also contains internal state for the SMC
## implementation
"""
    SMCIO{Particle, ParticleScratch}
Structs of this type should be constructed using the provided constructor.
Important fields:
- ```N::Int64``` Number of particles ``N``
- ```n::Int64``` Number of steps ``n``
- ```zetas::Vector{Particle}``` Time n particles ``\\zeta_n^1, \\ldots, \\zeta_n^N``
- ```ws::Vector{Float64}``` Time n weights ``W_n^1, \\ldots, W_n^N``
- ```logZhats::Vector{Float64}``` ``\\log(\\hat{Z}^N_1), \\ldots, \\log(\\hat{Z}^N_n)``
- ```allZetas::Vector{Vector{Particle}}``` All the particles
- ```allWs::Vector{Vector{Float64}}``` All the weights
- ```allAs::Vector{Vector{Int64}}``` All the ancestor indices
- ```allEves::Vector{Vector{Int64}}``` All the Eve indices
"""
struct SMCIO{Particle, ParticleScratch}
  N::Int64
  n::Int64
  zetas::Vector{Particle}
  ws::Vector{Float64}
  logZhats::Vector{Float64}

  internal::_SMCInternal{Particle, ParticleScratch} # for internal use only

  allZetas::Vector{Vector{Particle}}
  allWs::Vector{Vector{Float64}}
  allAs::Vector{Vector{Int64}}
end

"""
    SMCIO{Particle, ParticleScratch}(N::Int64, n::Int64) where {Particle, ParticleScratch}
Constructor for ```SMCIO``` structs.
"""
function SMCIO{Particle, ParticleScratch}(N::Int64, n::Int64) where
  {Particle, ParticleScratch}
  @assert hasmethod(Particle, ()) "Particle() must exist"
  @assert hasmethod(ParticleScratch, ()) "ParticleScratch() must exist"

  zetas = Vector{Particle}(undef, N)
  for i in 1:N
    zetas[i] = Particle()
  end

  ws = Vector{Float64}(undef, N)
  logZhats = Vector{Float64}(undef, n)

  allZetas = Vector{Vector{Particle}}(undef, n)
  for i=1:n
    allZetas[i] = Vector{Particle}(undef, N)
    for j=1:N
      allZetas[i][j] = Particle()
    end
  end
  allWs = Vector{Vector{Float64}}(undef, n)
  for i=1:n
    allWs[i] = Vector{Float64}(undef, N)
  end
  allAs = Vector{Vector{Int64}}(undef, n-1)
  for i=1:n-1
    allAs[i] = Vector{Int64}(undef, N)
  end

  internal::_SMCInternal = _SMCInternal{Particle, ParticleScratch}(N, n)

  return SMCIO(N, n, zetas, ws, logZhats, internal, allZetas, allWs, allAs)
end
