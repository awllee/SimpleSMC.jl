import NonUniformRandomVariateGeneration: sampleSortedUniforms!,
  sampleCategorical, sampleCategoricalSorted!

# sample nans categorical r.v.s where Pr(X=i) ∝ p[i]
# required: length(Fs) = length(p), unifs[start+1:start+nans] are valid
@inline function _sampleCategoricalSorted!(vs::Vector{Int64}, p::Vector{Float64},
  nans::Int64, unifs::Vector{Float64}, Fs::Vector{Float64}, start::Int64,
  rng::RNG)
  sampleSortedUniforms!(unifs, start, nans, rng)
  cumsum!(Fs, p)
  @inbounds maxFs = Fs[length(p)]
  j = 1
  for i = 1:nans
    @inbounds while maxFs * unifs[start + i] > Fs[j]
      j += 1
    end
    @inbounds vs[start + i] = j
  end
end

@inline function _cresample!(smcio::SMCIO)
  smcio.internal.as[1] = 1
  _sampleCategoricalSorted!(smcio.internal.as, smcio.ws, smcio.N-1,
    smcio.internal.scratch1, smcio.internal.scratch2, 1, getRNG())
end

@inline function _resample!(smcio::SMCIO)
  sampleCategoricalSorted!(smcio.internal.as, smcio.ws,
    smcio.internal.scratch1, smcio.internal.scratch2, getRNG())
end
