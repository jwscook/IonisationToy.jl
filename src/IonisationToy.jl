module IonisationToy

struct Grid
  dx::Float64
  N::Int
  iondensity::Vector{Float64}
  neutraldensity::Vector{Float64}
end
function Grid(;L::Float64, N::Int,
    iondensityic::FionIC=(x)->1.0,
    neutraldensityic::FneuIC=(x)->0.0) where {FionIC,FneuIC}
  dx = L / N
  gridcentres = collect(range(0, stop=L, length=N) .+ L / 2N)
  return Grid(dx, N, iondensityic.(gridcentres), neutraldensityic.(gridcentres))
end
integraliondensity(g::Grid) = sum(g.iondensity) * g.dx
integralneutraldensity(g::Grid) = sum(g.neutraldensity) * g.dx

mutable struct Particle
  x::Float64
  vx::Float64
  weight::Float64
end

function particlesvector(g::Grid, nparticles, fvx, numberdensity)
  L = g.dx * g.N
  weight = numberdensity * L / nparticles
  return [Particle(rand() * L, fvx(), weight) for _ in 1:nparticles]
end

totalweight(ps::AbstractVector{Particle}) = sum(p.weight for p in ps)
integralparticledensity(ps::AbstractVector{Particle}, g::Grid) = totalweight(ps) / (g.dx * g.N)

cellid(p::Particle, g::Grid) = mod1(Int(ceil(p.x / g.dx)), g.N)

function pushparticles!(neutralparticles, grid; bc, dt::Float64=1.0)
  if bc == :periodic
    L = grid.dx * grid.N
    for p in neutralparticles
      p.x += p.vx * dt
      while p.x < 0; p.x += L; end
      while p.x > L; p.x -= L; end
    end
  end
end

function depositongrid!(g::Grid, neutralparticles)
  for p in neutralparticles
    g.neutraldensity[cellid(p, g)] += p.weight / g.dx
  end
end

function ionise!(g::Grid, neutralparticles, dt, Riz)
  fill!(g.neutraldensity, 0)
  depositongrid!(g, neutralparticles)
  for p in neutralparticles
    cid = cellid(p, g)
    deltaweight = clamp(Riz * g.iondensity[cid] * p.weight * dt, 0, p.weight)
    g.iondensity[cid] += deltaweight / g.dx
    p.weight -= deltaweight
  end
end

function simulation(grid::Grid, neutralparticles::Vector{Particle}; dt=1.0, Riz=1.0,
    iters=1000, checkpointevery=1, bc=:periodic)
  checkpoints = []
  push!(checkpoints, (deepcopy(grid),deepcopy(neutralparticles))) # push IC
  for i in 1:iters
    pushparticles!(neutralparticles, grid, dt=dt, bc=:periodic)
    ionise!(grid, neutralparticles, dt, Riz)
    (i % checkpointevery == 0) && push!(checkpoints, (deepcopy(grid),deepcopy(neutralparticles)))
  end
  return checkpoints
end

end # module IonisationToy
