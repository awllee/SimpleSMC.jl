"""
    smc!(model::SMCModel, smcio::SMCIO)
Run the SMC algorithm for the given model and input/output arguments.
"""
function smc!(model::SMCModel, smcio::SMCIO{Particle}) where Particle
  zetas = smcio.zetas
  zetaAncs = smcio.internal.zetaAncs
  lws = smcio.internal.lws
  ws = smcio.ws
  as = smcio.internal.as
  engine = getRNG()
  pScratch = smcio.internal.particleScratch

  logZhats = smcio.logZhats
  lZ = 0.0

  for p = 1:smcio.n
    p > 1 && _copyParticles!(zetaAncs, zetas, as)

    _mutateParticles!(zetas, engine, p, model.M!, zetaAncs, pScratch)
    _logWeightParticles!(lws, p, model.lG, zetas, pScratch)

    maxlw::Float64 = maximum(lws)
    ws .= exp.(lws .- maxlw)
    mws::Float64 = mean(ws)
    lZ += maxlw + log(mws)
    ws ./= mws
    @inbounds logZhats[p] = lZ

    p < smcio.n && _resample!(smcio)

    _intermediateOutput!(smcio, p)
  end
end

function _csmc!(model::SMCModel, smcio::SMCIO{Particle},
  ref::Vector{Particle}, refout::Vector{Particle}) where Particle
  zetas = smcio.zetas
  zetaAncs = smcio.internal.zetaAncs
  lws = smcio.internal.lws
  ws = smcio.ws
  as = smcio.internal.as
  engine = getRNG()
  pScratch = smcio.internal.particleScratch

  logZhats = smcio.logZhats
  lZ = 0.0

  for p = 1:smcio.n
    p > 1 && _copyParticles!(zetaAncs, zetas, as)

    _mutateParticles!(zetas, engine, p, model.M!, zetaAncs, pScratch, ref[p])
    _logWeightParticles!(lws, p, model.lG, zetas, pScratch)

    maxlw::Float64 = maximum(lws)
    ws .= exp.(lws .- maxlw)
    mws::Float64 = mean(ws)
    lZ += maxlw + log(mws)
    ws ./= mws
    @inbounds logZhats[p] = lZ

    p < smcio.n && _cresample!(smcio)

    _intermediateOutput!(smcio, p)
  end
end

"""
    csmc!(model::SMCModel, smcio::SMCIO, ref::Vector{Particle}, refout::Vector{Particle})

Run the conditional SMC algorithm for the given model, input/output arguments,
reference path and output path.

It is permitted for ref and refout to be the same.
"""
function csmc!(model::SMCModel, smcio::SMCIO{Particle},
  ref::Vector{Particle}, refout::Vector{Particle}) where Particle

  _csmc!(model, smcio, ref, refout)

  pickParticle!(refout, smcio)
end

"""
    csmc!(model::SMCModel, lM::F, smcio::SMCIO, ref::Vector{Particle}, refout::Vector{Particle})

Run the conditional SMC algorithm for the given model, input/output arguments,
reference path and output path. Backward samping is used to choose the new
reference path, which requires the provision of the lM function.

It is permitted for ref and refout to be the same.
"""
function csmc!(model::SMCModel, lM::F, smcio::SMCIO{Particle},
  ref::Vector{Particle}, refout::Vector{Particle}) where {F<:Function, Particle}

  _csmc!(model, smcio, ref, refout)

  pickParticleBS!(refout, smcio, lM)
end
