module IonisationToy

using Polyester, LoopVectorization, Base.Threads

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
  vy::Float64
  vz::Float64
  weight::Float64
end

function particlesvector(g::Grid, nparticles, fvx, numberdensity)
  L = g.dx * g.N
  weight = numberdensity * L / nparticles
  return [Particle(rand() * L, fvx(), 0.0, 0.0, weight) for _ in 1:nparticles]
end

totalweight(ps::AbstractVector{Particle}) = isempty(ps) ? 0.0 : sum(p.weight for p in ps)
integralparticledensity(ps::AbstractVector{Particle}, g::Grid) = totalweight(ps) / (g.dx * g.N)

cellid(p::Particle, g::Grid) = mod1(Int(ceil(p.x / g.dx)), g.N)

function pushparticles!(neutralparticles, grid; bc, dt::Float64=1.0)
  L = grid.dx * grid.N
  if bc == :periodic
    @threads for p in neutralparticles
      p.x += p.vx * dt
      while p.x < 0; p.x += L; end
      while p.x > L; p.x -= L; end
    end
  elseif bc == :dirichlet
    deletes = Int[]
    for (i, p) in enumerate(neutralparticles)
      p.x += p.vx * dt
      (0 <= p.x < L) || push!(deletes, i)
    end
    deleteat!(neutralparticles, deletes)
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
  deletes = Int[]
  for (i, p) in enumerate(neutralparticles)
    cid = cellid(p, g)
    # numerical under-representation of ionisation if clamp is required to
    # prevent particle weight going negative
    deltaweight = clamp(Riz * g.iondensity[cid] * p.weight * dt, 0, p.weight)
    g.iondensity[cid] += deltaweight / g.dx
    p.weight -= deltaweight
    p.weight <= 0 && push!(deletes, i) # careful of floating point
  end
  deleteat!(neutralparticles, deletes)
end

function sink!(g::Grid, bulkionsinkrate, dt)
  iszero(bulkionsinkrate) && return nothing
  @batch per=core for i in eachindex(g.iondensity)
    g.iondensity[i] -= min(bulkionsinkrate * dt, g.iondensity[i])
  end
  return nothing
end

function externalsources!(neutralparticles, grid, t, dt; vth, sourcedensity,
    npartperbc=length(neutralparticles)/grid.N, 
    leftboundarysource::L=nothing, rightboundarysource::R=nothing) where {L, R}
  smax = 6 * vth
  spatialextent = smax * dt
  for (f, side) in ((leftboundarysource, :left), (rightboundarysource, :right))
    isnothing(f) && continue
    direction = side == :left ? 1 : -1
    @assert abs(direction) == 1 # this is a sign only, so +1 or -1
    startingpoint = direction == 1 ? 0.0 : grid.dx * grid.N
    nnp0 = length(neutralparticles)
    sizehint!(neutralparticles, nnp0 + npartperbc)
    nnew = 0
    while nnew < npartperbc
      rx = startingpoint - direction * rand() * spatialextent
      # Only sample the half of vx space that sends particles into domain
      rvx = rand() * smax * direction
      # Boundary is only in x, so vy and vz not affected
      rvy = (rand() - 0.5) * 2 * smax
      rvz = (rand() - 0.5) * 2 * smax
      if rand() <= f(startingpoint, rvx, rvy, rvz, t)
         nnew += 1
         push!(neutralparticles, Particle(rx, rvx, rvy, rvz, -1.0))
      end
    end
    # factor of half is because it's an integral over half vx space
    weight = sourcedensity * spatialextent / npartperbc / 2
    for i in nnp0+1:length(neutralparticles)
      neutralparticles[i].weight = weight
    end
  end
  return neutralparticles
end

function simulation(grid::Grid, neutralparticles::Vector{Particle}; dt=1.0, Riz=1.0,
    iters=1000, checkpointevery=1, bc=:periodic, vth=1.0, sourcedensity=1.0,
    bulkionsinkrate=0.0, npartperbc = length(neutralparticles)/grid.N,
    leftboundarysource::L=nothing, rightboundarysource::R=nothing) where {L, R}
  checkpoints = []
  push!(checkpoints, (deepcopy(grid),deepcopy(neutralparticles))) # push IC
  for i in 1:iters
    externalsources!(neutralparticles, grid, i * dt, dt;
      vth=vth, sourcedensity=sourcedensity, npartperbc=npartperbc,
      leftboundarysource=leftboundarysource, rightboundarysource=rightboundarysource)
    pushparticles!(neutralparticles, grid, dt=dt, bc=bc)
    ionise!(grid, neutralparticles, dt, Riz)
    sink!(grid, bulkionsinkrate, dt)
    (i % checkpointevery == 0) && push!(checkpoints, (deepcopy(grid),deepcopy(neutralparticles)))
    (i % checkpointevery == 0) && @info "iteration $i out of $iters."
  end
  return checkpoints
end

end # module IonisationToy
