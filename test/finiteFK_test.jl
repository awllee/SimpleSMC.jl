using SimpleSMC
using RNGPool
import SequentialMonteCarlo.SMCModel
using SMCExamples.FiniteFeynmanKac
import SMCExamples.Particles.Int64Particle
using Compat.Test
using Compat.Random

function Id(p::Int64Particle)
  return p.x
end

function One(p::Int64Particle)
  return 1.0
end

function IdMinus2(p::Int64Particle)
  return p.x - 2
end

function testEtas(smcio, ffkout, verbose::Bool)
  testapproxequal(
    SimpleSMC.allEtas(smcio, Id, false),
    FiniteFeynmanKac.allEtas(ffkout, Id, false), 0.01, verbose)
  testapproxequal(
    SimpleSMC.allEtas(smcio, Id, true),
    FiniteFeynmanKac.allEtas(ffkout, Id, true), 0.01, verbose)
end

function testZs(smcio, ffkout, verbose::Bool)
  testapproxequal(smcio.logZhats, ffkout.logZhats, 0.1, verbose)
end


function testapproxequal(x::Vector{Tuple{Bool,Float64}},
  y::Vector{Tuple{Bool,Float64}}, tol::Float64, verbose::Bool)
  xNonneg = (p->p[1]).(x)
  xLogVals = (p->p[2]).(x)
  yNonneg = (p->p[1]).(y)
  yLogVals = (p->p[2]).(y)
  @test xNonneg == yNonneg
  testapproxequal(xLogVals, yLogVals, tol, verbose)
end

function testGammas(smcio, ffkout, verbose::Bool)
  testapproxequal(SimpleSMC.allGammas(smcio, IdMinus2, false),
    FiniteFeynmanKac.allGammas(ffkout, IdMinus2, false), 0.1, verbose)
  testapproxequal(SimpleSMC.allGammas(smcio, IdMinus2, true),
    FiniteFeynmanKac.allGammas(ffkout, IdMinus2, true), 0.1, verbose)
end

function testFullOutput(ffk::FiniteFeynmanKac.FiniteFK, verbose::Bool)
  ffkout = FiniteFeynmanKac.calculateEtasZs(ffk)
  model = FiniteFeynmanKac.makeSMCModel(ffk)
  n = model.maxn
  smcio = SMCIO{model.particle, model.pScratch}(2^20, n)
  smc!(model, smcio)

  @testset "Finite FK: eta" begin
    testEtas(smcio, ffkout, verbose)
  end

  @testset "Finite FK: Zhat" begin
    testZs(smcio, ffkout, verbose)
  end

  @testset "Finite FK: slgamma" begin
    testGammas(smcio, ffkout, verbose)
  end

end

function _getFreqs(model::SMCModel, smcio::SMCIO, m::Int64, d::Int64)
  v::Vector{Int64Particle} = FiniteFeynmanKac.Int642Path(1, d, smcio.n)
  counts = zeros(Int64, d^smcio.n)
  result = Vector{Float64}(undef, d^smcio.n)
  for i = 1:m
    csmc!(model, smcio, v, v)
    counts[FiniteFeynmanKac.Path2Int64(v, d)] += 1
  end
  result .= counts ./ m
  return result
end

function _getFreqs(model::SMCModel, smcio::SMCIO, m::Int64, d::Int64, lM)
  v::Vector{Int64Particle} = FiniteFeynmanKac.Int642Path(1, d, smcio.n)
  counts = zeros(Int64, d^smcio.n)
  result = Vector{Float64}(undef, d^smcio.n)
  for i = 1:m
    csmc!(model, lM, smcio, v, v)
    counts[FiniteFeynmanKac.Path2Int64(v, d)] += 1
  end
  result .= counts ./ m
  return result
end

function testcsmc()
  d = 3
  n = 3
  ffk = FiniteFeynmanKac.randomFiniteFK(d, n)

  model = FiniteFeynmanKac.makeSMCModel(ffk)
  densities = Vector{Float64}(undef, d^n)
  for i = 1:length(densities)
    densities[i] = FiniteFeynmanKac.fullDensity(ffk, FiniteFeynmanKac.Int642Path(i, d, n))
  end
  densities ./= sum(densities)

  nsamples = 2^14

  smcio = SMCIO{model.particle, model.pScratch}(4, model.maxn)
  freqs = _getFreqs(model, smcio, nsamples, d)
  testapproxequal(freqs, densities, 0.05, false)

  lM = FiniteFeynmanKac.makelM(ffk)
  freqs = _getFreqs(model, smcio, nsamples, d, lM)
  testapproxequal(freqs, densities, 0.05, false)
end

setRNGs(0)

verbose = false

d = 3
ffk = FiniteFeynmanKac.randomFiniteFK(d, 10)

@time testFullOutput(ffk, verbose)

@time @testset "Finite FK: csmc" begin
  testcsmc()
end
