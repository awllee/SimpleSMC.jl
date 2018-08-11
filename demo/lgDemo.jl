using SimpleSMC
import SequentialMonteCarlo.SMCModel
using RNGPool
using SMCExamples.LinearGaussian
using Random

include("test.jl")

setRNGs(0)

model, theta, ys, ko = LinearGaussian.defaultLGModel(10)
println(ko.logZhats)

numParticles = 1024
numTrials = 2

## just run the algorithm a few times
testSMC(model, numParticles, numTrials)

testCSMC(model, numParticles, numTrials)

lM = LinearGaussian.makelM(theta)
testCSMCBS(model, lM, numParticles, numTrials)
