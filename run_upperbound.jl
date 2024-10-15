import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
using Statistics, POMDPModels

include("BIB/BIB.jl")
using .BIB
include("Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt

##################################################################
#                     Parsing Arguments
##################################################################

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        required = true
    "--precision"
        help = "Precision parameter of SARSOP."
        arg_type = Float64
        default = 1e-2
    "--timeout", "-t"
        help = "Time untill timeout."
        arg_type = Float64
        default = -1.0
    "--path"
        help = "File path for data output."
        default = "Data/UpperBounds/"
    "--filename"
        help = "Filename (default: generated automatically)"
        default = ""
    "--solvers"
        help = "Solver to be run. Availble options: FIB, BIB, EBIB, WBIB, SARSOP, BIBSARSOP, EBIBSARSOP. (default: run all but BIBSARSOP & EBIBSARSOP)"
        default = "All"
    "--discount"
        help = "Discount factor"
        arg_type = Float64
        default = 0.95
    "--precompile"
        help = "Option to precomile all code by running at low horizon. Particularly relevant for small environments. (default: true)"
        arg_type = Bool 
        default = true
end

parsed_args = parse_args(ARGS, s)
env_name = parsed_args["env"]
timeout = parsed_args["timeout"]
path = parsed_args["path"]
filename = parsed_args["filename"]
solver_names = [parsed_args["solvers"]]
solver_names == ["All"] && (solver_names = ["BIB", "EBIB", "WBIB", "FIB", "SARSOP"])
discount = parsed_args["discount"]
discount_str = string(discount)[3:end]
precompile = parsed_args["precompile"]

if timeout == -1.0
	discount == 0.95 && (timeout = 1200.0)
	discount == 0.99 && (timeout = 1200.0)
end

##################################################################
#                       Defining Solvers 
##################################################################

solvers, solverargs, precomp_solverargs = [], [], []
SARSOPprecision = 1e-2
heuristicprecision, heuristicsteps = 1e-4, 1_000
discount == 0.95 && (heuristicprecision = 1e-3;  heuristicsteps = 250)
discount == 0.99 && (heuristicprecision = 1e-4;  heuristicsteps = 1_000)

timeout_sarsop = 1200.0

if "BIB" in solver_names
    push!(solvers, SBIBSolver)
    push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "EBIB" in solver_names
    push!(solvers, EBIBSolver)
    push!(solverargs, (name="BIBSolver (entropy)", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))    
end
if "WBIB" in solver_names
    push!(solvers, WBIBSolver)
    push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "FIB" in solver_names
    push!(solvers, FIBSolver_alt)
    push!(solverargs, (name="FIB", sargs=(max_iterations=heuristicsteps*4, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "SARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=heuristicsteps*4, precision=heuristicprecision)
    push!(solverargs, (name="SARSOP", sargs=(precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
    precomp_h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=2)
    push!(precomp_solverargs, ( sargs=(max_its=1, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "BIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.SBIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="BIB-SARSOP", sargs=( precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "EBIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.EBIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="EBIB-SARSOP", sargs=( precision=precision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
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

if env_name == "ABC"
    include("Environments/ABCModel.jl"); using .ABCModel
    abcmodel = SparseTabularPOMDP(ABC(discount=discount))
    push!(envs, abcmodel)
    push!(envargs, (name="ABCModel",))
    ### Tiger
end
if env_name == "Tiger"
    tiger = POMDPModels.TigerPOMDP()
    tiger.discount_factor = discount
    push!(envs, SparseTabularPOMDP(tiger))
    push!(envargs, (name="Tiger",))
    ### RockSample
end
if env_name == "RockSample5"
    map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
    rocksamplesmall = RockSample.RockSamplePOMDP(map_size, rock_pos)
    push!(envargs, (name="RockSample ()",))
    push!(envs, rocksamplesmall)
end
# if env_name == "RockSample10"
#     map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
#     rocksample10 = RockSample.RockSamplePOMDP(map_size, rock_pos)
#     push!(envargs, (name="RockSample (10)",))
#     push!(envs, rocksamplelarge)
# end
# if env_name == "RockSample11"
#     map_size, rock_pos = (11,11), [(1,2), (2,7), (3,9), (4,2), (5,7), (5,10), (7,4), (8,8), (10,4) ] # Bigger Boy!
#     rocksample11 = RockSample.RockSamplePOMDP(map_size, rock_pos)
#     push!(envargs, (name="RockSample (11)",))
#     push!(envs, rocksample11)
# end
if env_name == "RockSample7"
    map_size, rock_pos = (7,7), [(1,2), (2,6), (3,3), (3,4), (4,7),(6,1),(6,4),(7,3)] # HSVI setting!
    rocksamplelarge = RockSample.RockSamplePOMDP(map_size, rock_pos)
    push!(envargs, (name="RockSample (7,8)",))
    push!(envs, rocksamplelarge)
end
if env_name == "K-out-of-N2"
    # ### K-out-of-N
    k_model2 = K_out_of_N(N=2, K=2, discount=discount)
    push!(envs, SparseTabularPOMDP(k_model2))
    push!(envargs, (name="K-out-of-N (2)",))
end
if env_name == "K-out-of-N3"
    k_model3 = K_out_of_N(N=3, K=3, discount=discount)
    push!(envs, SparseTabularPOMDP(k_model3))
    push!(envargs, (name="K-out-of-N (3)",))
end
if env_name == "FrozenLake4"
    # Frozen Lake esque
    lakesmall = FrozenLakeSmall
    lakesmall.discount = discount
    push!(envs, SparseTabularPOMDP(lakesmall))
    push!(envargs, (name="Frozen Lake (4)",))
end
if env_name == "FrozenLake10"
    lakelarge = FrozenLakeLarge
    lakelarge.discount = discount
    push!(envs, SparseTabularPOMDP(lakelarge))
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
    tigergrid = SparseTigerGrid(discount=discount)
    push!(envs, tigergrid)
    push!(envargs, (name="TigerGrid",))
end
if env_name == "aloha10"
    aloha10 = Sparse_aloha10(discount=discount)
    push!(envs, aloha10)
    push!(envargs, (name="aloha10",))
end
if env_name == "aloha30"
    aloha30 = Sparse_aloha30(discount=discount)
    push!(envs, aloha30)
    push!(envargs, (name="aloha30",))
end
if env_name == "baseball"
    baseball = Sparse_baseball(discount=discount)
    push!(envs, baseball)
    push!(envargs, (name="baseball",))
end
if env_name == "cit"
    cit = Sparse_cit(discount=discount)
    push!(envs, cit)
    push!(envargs, (name="cit",))
end
if env_name == "fourth"
    fourth = Sparse_fourth(discount=discount)
    push!(envs, fourth)
    push!(envargs, (name="fourth",))
end
if env_name == "mit"
    mit = Sparse_mit(discount=discount)
    push!(envs, mit)
    push!(envargs, (name="mit",))
end
if env_name == "pentagon"
    pentagon = Sparse_pentagon(discount=discount)
    push!(envs, pentagon)
    push!(envargs, (name="pentagon",))
end
if env_name == "sunysb"
    sunysb = Sparse_sunysb(discount=discount)
    push!(envs, sunysb)
    push!(envargs, (name="sunysb",))
end
if env_name == "grid"
    Grid = Sparse_Grid(discount=discount)
    push!(envs, Grid)
    push!(envargs, (name="Grid",))
end
if env_name == "Tag"
    ### Tag
    using TagPOMDPProblem
    tag = TagPOMDPProblem.TagPOMDP(discount_factor=discount)
    push!(envs, tag)
    push!(envargs, (name="Tag",))
end

##################################################################
#                           Run 
##################################################################

sims, steps = 1_000, 1_000

policy_names = map(sarg -> sarg.name, solverargs)
env_names = map(envarg -> envarg.name, envargs)
nr_pols, nr_envs = length(policy_names), length(env_names)

upperbounds_init = zeros(nr_envs, nr_pols)
upperbounds_sampled = zeros( nr_envs, nr_pols)
return_means = zeros( nr_envs, nr_pols)
time_solve = zeros( nr_envs, nr_pols)
time_online = zeros( nr_envs, nr_pols)

for (m_idx,(model, modelargs)) in enumerate(zip(envs, envargs))
    for (s_idx,(solver, solverarg)) in enumerate(zip(solvers, solverargs))
        # Calculate & print model size
        # model = SparseTabularPOMDP(model) #breaks RockSample...
        constants = BIB.get_constants(model)
        SAO_probs, SAOs = BIB.get_all_obs_probs(model; constants)
        B, B_idx = BIB.get_belief_set(model, SAOs; constants)
	Br = BIB.get_Br(model, B, constants)
	Data = BIB.BIB_Data(zeros(2,2), B, B_idx,Br, SAO_probs, SAOs, Dict(zip(constants.S, 1:constants.ns)), constants)
        BBao_data = BIB.get_Bbao(model, Data, constants)
        env_data = Dict(
            "ns" => constants.ns,
            "na" => constants.na,
            "no" => constants.no,
            "nb" => length(B),
            "nbao"=> length(BBao_data.Bbao) + length(B),
            "discount"=> discount
	    )

        # Precompile
        if precompile
            precomp_solver = solver(;precomp_solverargs[s_idx].sargs...)
            _t = @elapsed begin
                _p, _i = POMDPTools.solve_info(precomp_solver, model; precomp_solverargs[s_idx].pargs...) 
            end
        end

        # Compute policy & get upper bound
        solver = solver(;solverarg.sargs...)
        t = @elapsed begin
            policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...) 
        end
        if (info isa Nothing)
            ub = POMDPs.value(policy, POMDPs.initialstate(model))
            lb = -1
        else
            ub = info.ub
            lb =  info.lb
        end       

        # Simulate policy & get avg returns
        #rs = []
        #t0_sims = time()
        #for i=1:sims
        #    rtot = 0
        #    for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
        #        rtot += POMDPs.discount(model)^(t-1) * r
        #    end
        #    push!(rs,rtot)
        #end
        #t_sims = time() - t0_sims
        #rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
	rs_avg, rs_min, rs_max = -1.0, -1.0, -1.0
        t_sims = -1.0
	# Writing data to files
        data_dict = Dict(
            "env" => env_name,
            "env_full" => modelargs.name,
            "env_data" => env_data,
            "solver" => solverarg.name,
            # Solving data
            "solvetime" => t,
            "ub" => ub,
            "lb" => lb,
            # Simulation data
            "simtime" => t_sims,
            "ravg" => rs_avg
        )
        json_str = JSON.json(data_dict)
        println(env_name, " ", t, " ", ub)
        if filename == ""
		thisfilename =  path * "UpperBoundTest_$(env_name)_$(solver_names[s_idx])_d$(discount_str).json"
        else
            thisfilename = path * filename * solverarg.name
        end
        open(thisfilename, "w") do file
            write(file, json_str)
        end
    end
end
