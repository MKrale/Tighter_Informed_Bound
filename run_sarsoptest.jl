import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
include("BIB.jl")
using .BIB
using Statistics, POMDPModels

##################################################################
#                     Parsing Arguments
##################################################################

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        required = true
    "--timeout", "-t"
        help = "Time untill timeout."
        arg_type = Float64
        default = 60.0
    "--precision"
        help = "Precision parameter of SARSOP."
        arg_type= Float64
        default = 1e-2
    "--path"
        help = "File path for data output."
        default = "./Data/"
    "--filename"
        help = "Filename (default: generated automatically)"
        default = ""
    "--solvers"
        help = "Solver to be run. Availble options: standard, BIB, EBIB. (default: run all)"
        default = ""
    "--discount"
        help = "Discount factor"
        arg_type = Float64
        default = 0.95
end

parsed_args = parse_args(ARGS, s)
timeout = parsed_args["timeout"]
env_name = parsed_args["env"]
precision = parsed_args["precision"]
path = parsed_args["path"]
filename = parsed_args["filename"]
solver_name = parsed_args["solvers"]
discount = parsed_args["discount"]
discount_str = string(discount)[3:end]

##################################################################
#                       Defining Solvers 
##################################################################

solvers, solverargs = [], []
include("Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt

if solver_name in  ["standard", ""]
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    push!(solverargs, (name="SARSOP", sargs=(precision=precision, max_time=timeout, verbose=false), pargs=()))
end
if solver_name in  ["BIB", ""]
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = SBIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="BIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if solver_name in  ["EBIB", ""]
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = EBIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="EBIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver), pargs=()))
end

##################################################################
#                       Selecting env 
##################################################################

import RockSample
# This env is very difficult to work with for some reason...
POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
POMDPs.discount(M::RockSample.RockSamplePOMDP) = discount
include("Environments/K-out-of-N.jl"); using .K_out_of_Ns
include("Environments/CustomGridworld.jl"); using .CustomGridWorlds
include("Environments/Sparse_models/SparseModels.jl"); using .SparseModels



envs, envargs = [], []

    ###ABC
if env_name == "ABC"
    include("Environments/ABCModel.jl"); using .ABCModel
    abcmodel = ABC(discount=discount)
    push!(envs, abcmodel)
    push!(envargs, (name="ABCModel",))
    ### Tiger
end
if env_name == "Tiger"
    tiger = POMDPModels.TigerPOMDP()
    tiger.discount_factor = discount
    push!(envs, tiger)
    push!(envargs, (name="Tiger",))
    ### RockSample
end
if env_name == "RockSample5"
    map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
    rocksamplesmall = RockSample.RockSamplePOMDP(map_size, rock_pos)
    push!(envargs, (name="RockSample ()",))
    push!(envs, rocksamplesmall)
end
if env_name == "RockSample10"
    map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
    rocksamplelarge = RockSample.RockSamplePOMDP(map_size, rock_pos)
    push!(envargs, (name="RockSample (10)",))
    push!(envs, rocksamplelarge)
end
if env_name == "K-out-of-N2"
    # ### K-out-of-N
    k_model2 = K_out_of_N(N=2, K=2, discount=discount)
    push!(envs, k_model2)
    push!(envargs, (name="K-out-of-N (2)",))
end
if env_name == "K-out-of-N3"
    k_model3 = K_out_of_N(N=3, K=3, discount=discount)
    push!(envs, k_model3)
    push!(envargs, (name="K-out-of-N (3)",))
end
if env_name == "FrozenLake4"
    # Frozen Lake esque
    lakesmall = FrozenLakeSmall
    lakesmall.discount = discount
    push!(envs, lakesmall)
    push!(envargs, (name="Frozen Lake (4)",))
end
if env_name == "FrozenLake10"
    lakelarge = FrozenLakeLarge
    lakelarge.discount = discount
    push!(envs, lakelarge)
    push!(envargs, (name="Frozen Lake (10)",))
end
if env_name == "Hallway1"
    hallway1 = Hallway1
    hallway.discount = discount
    push!(envs, hallway1)
    push!(envargs, (name="Hallway1",))
end
if env_name == "Hallway2"
    hallway2 = Hallway2
    hallway2.discount = discount
    push!(envs, hallway2)
    push!(envargs, (name="Hallway2",))
end
if env_name == "MiniHallway"
    minihall = CustomMiniHallway
    minihall.discount = discount
    push!(envs, minihall)
    push!(envargs, (name="MiniHallway",))
end
if env_name == "TigerGrid"
    tigergrid = TigerGrid
    tigergrid.discount = discount
    push!(envs, tigergrid)
    push!(envargs, (name="TigerGrid",))
end
if env_name == "SparseHallway1"
    hallway1 = SparseHallway1(discount=discount)
    push!(envs, hallway1)
    push!(envargs, (name="SparseHallway1",))
end
if env_name == "SparseHallway2"
    hallway2 = SparseHallway2(discount=discount)
    push!(envs, hallway2)
    push!(envargs, (name="SparseHallway2",))
end
if env_name == "SparseTigerGrid"
    tigergrid = TigerGrid(discount=discount)
    push!(envs, tigergrid)
    push!(envargs, (name="TigerGrid",))
end
if env_name == "Tag"
    ### Tag
    using TagPOMDPProblem
    tag = TagPOMDPProblem.TagPOMDP(discount_factor=discount)
    push!(envs, tag)
    push!(envargs, (name="Tag",))
end

env, env_arg = envs[1], envargs[1]


### Unused envs:

# ### DroneSurveilance
# import DroneSurveillance
# dronesurv = DroneSurveillance.DroneSurveillancePOMDP()
# push!(envs, dronesurv)
# push!(envargs, (name="DroneSurveilance",))

# ### Mini Hallway
# minihall = POMDPModels.MiniHallway()
# push!(envs, minihall)
# push!(envargs, (name="MiniHallway",))

# ### TMaze (Does not work with FIB)
# tmaze = POMDPModels.TMaze()
# POMDPs.reward(tmaze::TMaze, s::POMDPTools.ModelTools.TerminalState,a ) = 0
# push!(envs, tmaze)
# push!(envargs, (name="TMaze",))

# # For some reason, the envs below do not work:

# ## SubHunt (No 'observations' defined)
# import SubHunt
# subhunt = SubHunt.SubHuntPOMDP()
# push!(envs, subhunt)
# push!(envargs, (name="SubHunt",))

# ### LaserTag (No 'observations' defined)
# import LaserTag
# lasertag = LaserTag.gen_lasertag()
# push!(envs, lasertag)
# push!(envargs, (name="LaserTag",))



##################################################################
#                           Run 
##################################################################


ubs, lbs = Tuple{Vector{Float64}, Vector{Float64}}[], Tuple{Vector{Float64}, Vector{Float64}}[]

for (i, (solver, solverarg)) in enumerate(zip(solvers, solverargs))
    
    solver = solver(;solverarg.sargs...)
    policy, info = POMDPTools.solve_info(solver, env; solverarg.pargs...)
    
    data_dict = Dict(
        "env" => env_name,
        "env_full" => env_arg.name,
        "solver" => solverarg.name,
        "timeout" => info.timeout,
        "runtime" => last(info.times),
        "final_ub" => last(info.ubs),
        "final_lw" => last(info.lbs),
        "ubs" => info.ubs,
        "lbs" => info.lbs,
        "times" => info.times
    )

    json_str = JSON.json(data_dict)
    if filename == ""
        thisfilename =  path * "Sarsoptest_$(env_name)_$(solverarg.name)_t$(Int(ceil(timeout)))_d$discount_str.json"
    else
        thisfilename = path * filename * solverarg.name
    end
    open(thisfilename, "w") do file
        write(file, json_str)
    end
end






