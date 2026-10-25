using Random
using SimpleSMC
import SequentialMonteCarlo: SMCModel

mutable struct RNGTestParticle
  x::Float64
  RNGTestParticle() = new(0.0)
end

@testset "Task-local RNG compatibility" begin
  function mutate!(dest, rng::AbstractRNG, p, ancestor, scratch)
    yield()
    dest.x = (p == 1 ? 0.0 : ancestor.x) + randn(rng)
  end
  model = SMCModel(mutate!, (p, x, scratch) -> 0.0, 3,
    RNGTestParticle, Nothing)
  function runseeded()
    Random.seed!(123)
    io = SMCIO{RNGTestParticle, Nothing}(64, 3)
    smc!(model, io)
    return io
  end
  first = fetch(Threads.@spawn runseeded())
  second = fetch(Threads.@spawn runseeded())
  @test first.logZhats == zeros(3)
  @test first.ws == ones(64)
  @test [x.x for x in first.zetas] == [x.x for x in second.zetas]
  @test all(isfinite(x.x) for x in first.zetas)
  @test all(issorted, first.allAs)

  for backward in (false, true)
    ref = [RNGTestParticle() for _ in 1:3]
    for x in ref
      x.x = 9.0
    end
    if backward
      lM = (p, ancestor, particle, scratch) -> 0.0
      fetch(Threads.@spawn csmc!(model, lM, first, ref, ref))
    else
      fetch(Threads.@spawn csmc!(model, first, ref, ref))
    end
    @test all(first.allZetas[p][1].x == 9.0 for p in 1:3)
    @test all(a[1] == 1 && all(1 .<= a .<= 64) for a in first.allAs)
    @test all(isfinite(x.x) for x in ref)
  end
end
