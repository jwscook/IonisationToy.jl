using IonisationToy, QuadGK, Test, LinearFitXYerrors

@testset "Integration Test 1" begin
  L = 1.0
  NG = 128
  nn0 = 11.0

  Riz = 1.0 # doesn't matter

  vth = 1# / 7
  dt = L / NG / vth# / 13
  tend = 20L / vth#2048 * 4
  @show iters = nextpow(2, Int(ceil(tend / dt)))
  @show checkpointevery = max(1, iters ÷ 128)

  fmaxwellian(x, vx, vy, vz, t) = exp(-(vx^2 + vy^2 + vz^2) / vth^2)

  grid = IonisationToy.Grid(;L=L, N = NG,
    iondensityic=x->0.0,
    neutraldensityic=x->0);

  neutralparticles = Vector{IonisationToy.Particle}();

  checkpoints = IonisationToy.simulation(grid, neutralparticles, iters=iters, dt=dt, Riz=Riz,
    bc=:dirichlet,
    checkpointevery=checkpointevery,
    leftboundarysource=fmaxwellian,
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
  fsol(x, v, t) = (1 - heaviside(x - v * t)) * nn0 * exp(-v^2 / vth^2) / sqrt(pi) / vth
  nt = [HCubature.hcubature(xv->fsol(xv..., t), (0, -8vth), (L, 8vth), rtol=1e-5)[1] for t in ts]
  
  @test isapprox(yp, yn)
  @test isapprox(nt, yp, rtol=1e-2, atol=nn0/10)
  @test isapprox(yp[1:3], ts[1:3] * nn0 * vth / 2sqrt(pi) / L, rtol=1e-2)

#heatmap(x, v, [fsol(xi, vi, 100L / vth) for vi in v, xi in x])
#plot(ts, nt)
#plot!(ts, yp)
#plot!(ts[1:10], ts[1:10] * nn0 * vth / 2sqrt(pi) / L)
#plot!([ts[1], ts[end]], [1, 1] .* nn0 / 2)

end
