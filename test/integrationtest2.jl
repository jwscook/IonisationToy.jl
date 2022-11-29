using IonisationToy, QuadGK, Test, LinearFitXYerrors, HCubature, Random
Random.seed!(0)

#@testset "Integration Test 2" begin
  L = 1.0
  NG = 128
  nn0 = 11.0

  Riz = 0.0 # doesn't matter

  vth = 1# / 7
  dt = L / NG / vth# / 13
  tend = 20L / vth#2048 * 4
  iters = nextpow(2, Int(ceil(tend / dt)))
  checkpointevery = max(1, iters ÷ 128)

  function fknudsencosineleft(x, vx, vy, vz, t)
    v = sqrt(vx^2 + vy^2 + vz^2)
    vdotn = vx # dot([vx, vy, vx], [1, 0, 0])
    iszero(v) && return 1.0
    return vx / v * exp(-v^2 / vth^2)
  end

  # if this fails then the test logic is wrong
  @assert isapprox(1, HCubature.hcubature(v->4/(sqrt(π) * vth)^3 * fknudsencosineleft(0, v..., 0),
    (0.0, -8vth, -8vth),
    (8vth, 8vth, 8vth), rtol=1e-5)[1], rtol=1e4)

  grid = IonisationToy.Grid(;L=L, N = NG,
    iondensityic=x->0.0,
    neutraldensityic=x->0);

  neutralparticles = Vector{IonisationToy.Particle}();

  checkpoints = IonisationToy.simulation(grid, neutralparticles, iters=iters, dt=dt, Riz=Riz,
    bc=:dirichlet,
    checkpointevery=checkpointevery,
    leftboundarysource=fknudsencosineleft,
    sourcedensity=nn0,
    npartperbc=512,
    vth=vth);

  ts = (0:length(checkpoints)-1) .* dt * checkpointevery
  grids = [checkpoints[i][1] for i in eachindex(ts)] 
  particles = [checkpoints[i][2] for i in eachindex(ts)] 

  yi = [IonisationToy.integraliondensity(grids[i]) for i in eachindex(ts)]
  yn = [IonisationToy.integralneutraldensity(grids[i]) for i in eachindex(ts)]
  yp = [IonisationToy.totalweight(particles[i]) for i in eachindex(ts)]

  heaviside(x) = x > 0
  function fsol(x, vx, vy, vz, t)
    return (1 - heaviside(x - vx * t)) * nn0 * 4/(sqrt(π) * vth)^3 * fknudsencosineleft(x, vx, vy, vz, t)
  end
  nt = [HCubature.hcubature(xv->fsol(xv..., t),
    (0, 0.0, -8vth, -8vth),
    (L, 8vth, 8vth, 8vth), rtol=1e-3)[1] for t in ts]
  
  @test isapprox(yp, yn)
  @test isapprox(nt, yp, rtol=1e-2, atol=nn0/10)

#heatmap(x, v, [fsol(xi, vi, 100L / vth) for vi in v, xi in x])
#plot(ts, nt)
#plot!(ts, yp)
#plot!(ts[1:10], ts[1:10] * nn0 * vth / 2sqrt(pi) / L)
#plot!([ts[1], ts[end]], [1, 1] .* nn0 / 2)

#end
