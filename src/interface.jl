@inline function _getZetas(smcio::SMCIO{Particle}, p::Int64) where Particle
  @assert 1 <= p <= smcio.n
  p == smcio.n && return smcio.zetas
  return smcio.allZetas[p]
end

@inline function _getWs(smcio::SMCIO, hat::Bool, p::Int64)
  @assert 1 <= p <= smcio.n
  if hat
    p == smcio.n && return smcio.ws
    return smcio.allWs[p]
  else
    return smcio.internal.vecOnes
  end
end

"""
    eta(smcio::SMCIO{Particle}, f::F, hat::Bool, p::Int64) where {Particle, F<:Function}

Compute:
  - ```!hat```: ``\\eta^N_p(f)``
  - ```hat```:  ``\\hat{\\eta}_p^N(f)``
"""
function eta(smcio::SMCIO{Particle}, f::F, hat::Bool, p::Int64) where
  {Particle, F<:Function}
  zetas::Vector{Particle} = _getZetas(smcio, p)
  ws::Vector{Float64} = _getWs(smcio, hat, p)
  v = f(zetas[1]) * ws[1]
  for i = 2:smcio.N
    @inbounds v += f(zetas[i]) * ws[i]
  end
  return v / smcio.N
end

"""
    allEtas(smcio::SMCIO, f::F, hat::Bool) where F<:Function

Compute ```eta(smcio::SMCIO, f::F, hat::Bool, p)``` for p in {1, …, smcio.n}
"""
function allEtas(smcio::SMCIO, f::F, hat::Bool) where F<:Function
  T = typeof(f(smcio.zetas[1]) / smcio.N)
  result::Vector{T} = Vector{T}(undef, smcio.n)
  for p = 1:smcio.n
    @inbounds result[p] = eta(smcio, f, hat, p)
  end
  return result
end

"""
    slgamma(smcio::SMCIO, f::F, hat::Bool, p::Int64) where {Particle, F<:Function}

Compute:
- ```!hat```: ``(\\eta^N_p(f) \\geq 0, \\log |\\gamma^N_p(f)|)``
- ```hat```:  ``(\\hat{\\eta}^N_p(f) \\geq 0, \\log |\\hat{\\gamma}_p^N(f)|)``
The result is returned as a ```Tuple{Bool, Float64}```: the first component
represents whether the returned value is non-negative, the second is the
logarithm of the absolute value of the approximation.
"""
function slgamma(smcio::SMCIO, f::F, hat::Bool, p::Int64) where
  F<:Function
  @assert 1 <= p <= smcio.n "p must be between 1 and smcio.n"
  idx::Int64 = p - 1 + hat
  logval = idx == 0 ? 0.0 : smcio.logZhats[idx]
  v::Float64 = eta(smcio, f, hat, p)
  logval += log(abs(v))
  return (v >= 0, logval)
end

"""
    allGammas(smcio::SMCIO, f::F, hat::Bool) where F<:Function

Compute ```slgamma(smcio::SMCIO, f::F, hat::Bool, p)``` for p in {1, …, smcio.n}
"""
function allGammas(smcio::SMCIO, f::F, hat::Bool) where F<:Function
  result::Vector{Tuple{Bool, Float64}} =
    Vector{Tuple{Bool, Float64}}(undef, smcio.n)
  for p = 1:smcio.n
    @inbounds result[p] = slgamma(smcio, f, hat, p)
  end
  return result
end
