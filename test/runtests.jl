using Test

@testset "IonisationToy tests" begin
@show "1"
include("./integrationtest0.jl")
@show "2"
include("./integrationtest1.jl")
@show "3"
include("./integrationtest2.jl")
end
