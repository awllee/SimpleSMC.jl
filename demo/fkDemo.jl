using SimpleSMC
import SequentialMonteCarlo.SMCModel
using RNGPool
using SMCExamples.FiniteFeynmanKac
using Random

include("test.jl")

setRNGs(0)

d = 3
ffk = FiniteFeynmanKac.randomFiniteFK(d, 10)
model = FiniteFeynmanKac.makeSMCModel(ffk)

numParticles = 1024
numTrials = 2

## just run the algorithm a few times
testSMC(model, numParticles, numTrials)

testCSMC(model, numParticles, numTrials)

lM = FiniteFeynmanKac.makelM(ffk)
testCSMCBS(model, lM, numParticles, numTrials)
