import Compat.undef

function testSMC(model::SMCModel, N::Int64, numTrials::Int64)
  smcio = SMCIO{model.particle, model.pScratch}(N, model.maxn)
  println("Running SMC. N = $N")
  for i=1:numTrials
    @time smc!(model, smcio)
    left = max(length(smcio.logZhats)-10,1)
    right = length(smcio.logZhats)
    println(smcio.logZhats[left:right])
  end
end

function testCSMC(model::SMCModel, N::Int64, numTrials::Int64)
  smcio = SMCIO{model.particle, model.pScratch}(N, model.maxn)
  println("Running CSMC. N = $N")
  smc!(model, smcio)
  v::Vector{model.particle} = Vector{model.particle}(undef, smcio.n)
  for i in 1:smcio.n
    v[i] = model.particle()
  end
  SimpleSMC.pickParticle!(v, smcio)
  for i=1:numTrials
    @time csmc!(model, smcio, v, v)
    left = max(length(smcio.logZhats)-10,1)
    right = length(smcio.logZhats)
    println(smcio.logZhats[left:right])
  end
end

function testCSMCBS(model::SMCModel, lM::F, N::Int64, numTrials::Int64) where F<:Function
  smcio = SMCIO{model.particle, model.pScratch}(N, model.maxn)
  println("Running CSMC-BS. N = $N")
  smc!(model, smcio)
  v::Vector{model.particle} = Vector{model.particle}(undef, smcio.n)
  for i in 1:smcio.n
    v[i] = model.particle()
  end
  SimpleSMC.pickParticle!(v, smcio)
  for i=1:numTrials
    @time csmc!(model, lM, smcio, v, v)
    left = max(length(smcio.logZhats)-10,1)
    right = length(smcio.logZhats)
    println(smcio.logZhats[left:right])
  end
end
