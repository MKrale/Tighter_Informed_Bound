module SparseModels

using POMDPs, POMDPModels, POMDPTools, SparseArrays

include("Hallway1.jl")
include("Hallway2.jl")
include("TigerGrid.jl")

export 
SparseHallway1, SparseHallway2, SparseTigerGrid

end