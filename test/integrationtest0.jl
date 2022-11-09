using IonisationToy, QuadGK, Test, LinearFitXYerrors

@testset "Integration Test 0" begin
  L = 1.0 + rand()
  NG = 128
  ni0 = 2 + 3 * rand()
  nn0 = 3 + 2 * rand()

  NP = 2^14
  Riz = 1/pi#0.5 + rand()

  iters = 1000
  dt = Riz * nn0 / 400

  neutraldensityic(x) = 0.0
  iondensityic(x) = ni0
  grid = IonisationToy.Grid(;L=L, N = NG,
                            iondensityic=iondensityic,
                            neutraldensityic=neutraldensityic);

  vxic() = randn()
  neutralparticles = IonisationToy.particlesvector(grid, NP, vxic, nn0)

  @test IonisationToy.integralneutraldensity(grid) ≈ 0.0
  IonisationToy.depositongrid!(grid, neutralparticles)
  @test IonisationToy.integralneutraldensity(grid) ≈ nn0 * L


  checkpoints = IonisationToy.simulation(grid, neutralparticles, iters=iters, dt=dt, Riz=Riz)

  t = (0:length(checkpoints)-1) .* dt
  grids = [checkpoints[i][1] for i in eachindex(t)] 
  particles = [checkpoints[i][2] for i in eachindex(t)] 

  yi = [IonisationToy.integraliondensity(grids[i]) for i in eachindex(t)]
  yn = [IonisationToy.integralneutraldensity(grids[i]) for i in eachindex(t)]
  yp = [IonisationToy.totalweight(particles[i]) for i in eachindex(t)]

  @test all(@. yi + yp ≈ L * (nn0 + ni0))
  @test all(isapprox(yp, yn, rtol=1e-1))
  # early time linear solution
  A = 1
  B = min(20, length(t))
  fit = linearfitxy(t[A:B], log.(yi[A:B]))
  slope = fit.b
  @test isapprox(slope, Riz * nn0, rtol=1e-1)

  # long time linear solution from particles
  B = length(t)
  A = B ÷ 2
  fit = linearfitxy(t[A:B], log.(yp[A:B]))
  slope = fit.b
  @test isapprox(slope, - Riz * (ni0 + nn0), rtol=1e-2)
end
