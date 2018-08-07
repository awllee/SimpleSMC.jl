# adapted from https://discourse.julialang.org/t/how-to-copy-all-fields-without-changing-the-referece/945/5
@inline @generated function particleCopy!(dest, src)
  fieldNames = fieldnames(dest)
  fieldTypes = dest.types
  numFields = length(fieldNames)
  expressions = Array{Expr}(undef, numFields)

  for i = 1:numFields
    fieldName = fieldNames[i]
    fieldType = fieldTypes[i]
    @assert !fieldType.mutable || hasmethod(copyto!, (fieldType,
      fieldType)) "$fieldName::$fieldType : copyto! must exist for mutable Particle fields"
    if fieldType.mutable
      @inbounds expressions[i] = :(copyto!(dest.$fieldName, src.$fieldName))
    else
      @inbounds expressions[i] = :(dest.$fieldName = src.$fieldName)
    end
  end
  body = Expr(:block, expressions...)

  quote
    $body
    return
  end
end

## better not to inline ; enough work should be done in the loop
function _copyParticles!(out::Vector{Particle}, in::Vector{Particle}) where Particle
  for i in eachindex(out)
    @inbounds particleCopy!(out[i], in[i])
  end
end

## better not to inline ; enough work should be done in the loop
function _copyParticles!(zetaAncs::Vector{Particle}, zetas::Vector{Particle},
  as::Vector{Int64}) where Particle
  for i in eachindex(zetaAncs)
    @inbounds particleCopy!(zetaAncs[i], zetas[as[i]])
  end
end

## better not to inline ; enough work should be done in the loop
function _mutateParticles!(zetas::Vector{Particle}, rng::RNG, p::Int64, M!::F,
  zetaAncs::Vector{Particle}, pScratch::ParticleScratch) where
  {Particle, F<:Function, ParticleScratch}
  for j in eachindex(zetas)
    @inbounds M!(zetas[j], rng, p, zetaAncs[j], pScratch)
  end
end

## better not to inline ; enough work should be done in the loop
function _mutateParticles!(zetas::Vector{Particle}, rng::RNG, p::Int64, M!::F,
  zetaAncs::Vector{Particle}, pScratch::ParticleScratch, xref::Particle) where {Particle,
  F<:Function, ParticleScratch}
  particleCopy!(zetas[1], xref)
  for j in 2:length(zetas)
    @inbounds M!(zetas[j], rng, p, zetaAncs[j], pScratch)
  end
end

## better not to inline ; enough work should be done in the loop
function _logWeightParticles!(lws::Vector{Float64}, p::Int64, lG::F,
  zetas::Vector{Particle}, pScratch::ParticleScratch) where
  {F<:Function, Particle, ParticleScratch}
  for j in eachindex(zetas)
    @inbounds lws[j] = lG(p, zetas[j], pScratch)
  end
end

@inline function _intermediateOutput!(smcio::SMCIO, p::Int64)
  _copyParticles!(smcio.allZetas[p], smcio.zetas)
  @inbounds smcio.allWs[p] .= smcio.ws
  @inbounds p != smcio.n && (smcio.allAs[p] .= smcio.internal.as)
end

@inline function pickParticle!(path::Vector{Particle}, smcio::SMCIO{Particle}) where Particle
  @assert length(path) == smcio.n
  n::Int64 = smcio.n
  allAs::Vector{Vector{Int64}} = smcio.allAs
  allZetas::Vector{Vector{Particle}} = smcio.allZetas
  k::Int64 = sampleCategorical(smcio.ws, getRNG())

  @inbounds particleCopy!(path[n], allZetas[n][k])
  for p = n-1:-1:1
    @inbounds k = allAs[p][k]
    @inbounds particleCopy!(path[p], allZetas[p][k])
  end
  return
end

@inline function pickParticleBS!(path::Vector{Particle},
  smcio::SMCIO{Particle, ParticleScratch}, lM::F) where {Particle,
  ParticleScratch, F<:Function}
  @assert length(path) == smcio.n
  n::Int64 = smcio.n
  N::Int64 = smcio.N
  allAs::Vector{Vector{Int64}} = smcio.allAs
  allZetas::Vector{Vector{Particle}} = smcio.allZetas
  allWs::Vector{Vector{Float64}} = smcio.allWs
  bws::Vector{Float64} = smcio.internal.scratch1
  pScratch::ParticleScratch = smcio.internal.particleScratch

  rng = getRNG()
  k::Int64 = sampleCategorical(smcio.ws, rng)

  @inbounds particleCopy!(path[n], allZetas[n][k])
  for p = n-1:-1:1
    @inbounds bws .= log.(allWs[p])
    for j in 1:N
      @inbounds bws[j] += lM(p+1, allZetas[p][j], path[p+1], pScratch)
    end
    m::Float64 = maximum(bws)
    bws .= exp.(bws .- m)
    k = sampleCategorical(bws, rng)
    @inbounds particleCopy!(path[p], allZetas[p][k])
  end
end
