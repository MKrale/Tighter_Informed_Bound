using POMDPs
using POMDPTools, POMDPFiles
include("BIB/BIB.jl")
using .BIB
using Statistics, POMDPModels
using SparseArrays
import RockSample
using Profile, PProf

##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs = [], []

iters, tol = 250, 1e-5


### FIB
using FIB
push!(solvers, FIBSolver_alt)
push!(solverargs, (name="FIB", sargs=(max_iterations=iters,precision=tol), pargs=(), get_Q0=true))

# ### BIB
#  push!(solvers, SBIBSolver)
#  push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### EBIB
# push!(solvers, EBIBSolver)
# push!(solverargs, (name="BIBSolver (entropy)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### WBIBs
# push!(solvers, WBIBSolver)
# push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=250, precision=1e-5), pargs=(), get_Q0=true))

# SARSOP
# include("Sarsop_altered/NativeSARSOP.jl")
# import .NativeSARSOP_alt
# max_time = 120.0

# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# solver = SBIBSolver(max_iterations=1_000, precision=1e-5)
# push!(solverargs, (name="SARSOP+BIB (max $max_time s)", sargs=( epsilon=1e-2, precision=1e-2, max_time=max_time, verbose=false, heuristic_solver=solver), pargs=(), get_Q0=true))


# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# push!(solverargs, (name="SARSOP (max $max_time s)", sargs=( epsilon=1e-2, precision=1e-2, max_time=max_time, verbose=false), pargs=(), get_Q0=true))

# ### SARSOP
# using NativeSARSOP
# push!(solvers, NativeSARSOP.SARSOPSolver)
# push!(solverargs, (name="SARSOP", sargs=(epsilon=0.5, precision=0.001, kappa=0.5, delta=1e-1, max_time=2.0, max_steps=1, verbose=true), pargs=(), get_Q0=true))

### SARSOP (Wrapped)
# using SARSOP
# push!(solvers, SARSOP.SARSOPSolver)
# push!(solverargs, (name="SARSOP (wrapped)", sargs=(timeout=60., precision=1e-5, fast=false), pargs=(), is_AMSolver=false))

# ## POMCP 
# using BasicPOMCP
# push!(solvers, POMCPSolver)
# push!(solverargs, (name="Basic POMCP", sargs=(tree_queries=1000, max_time=0.5), pargs=(),is_AMSolver=false))

### AdaOPS
# using AdaOPS
# push!(solvers, AdaOPSSolver)
# push!(solvernames, "AdaOPS")
# push!(solverargs, Dict())
# push!(policyargs, Dict())

### ARDESPOT
# using ARDESPOT
# push!(solvers, DESPOTSolver)
# push!(solvernames, "ARDESPOT")
# push!(solverargs, Dict())
# push!(policyargs, Dict())

##################################################################
#                           Set Envs 
##################################################################

envs, envargs = [], []
discount = 0.95

# # # ### ABC
# include("Environments/ABCModel.jl"); using .ABCModel
# abcmodel = ABC(discount=discount)
# push!(envs, abcmodel)
# push!(envargs, (name="ABCModel",))

 
#  # ### Tiger
#  tiger = POMDPModels.TigerPOMDP()
#  tiger.discount_factor = discount
#  push!(envs, tiger)
#  push!(envargs, (name="Tiger",))


# # ### RockSample
import RockSample
# # This env is very difficult to work with for some reason...
POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
POMDPs.discount(M::RockSample.RockSamplePOMDP) = discount

map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
rocksamplesmall = RockSample.RockSamplePOMDP(map_size, rock_pos)
push!(envargs, (name="RockSample (5x5)",))
push!(envs, rocksamplesmall)

# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
# rocksamplelarge = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envargs, (name="RockSample (10x10)",))
# push!(envs, rocksamplelarge)

# # ### K-out-of-N
include("Environments/K-out-of-N.jl"); using .K_out_of_Ns

# k_model2 = K_out_of_N(N=2, K=2, discount=discount)
# push!(envs, k_model2)
# push!(envargs, (name="K-out-of-N (2)",))

# k_model3 = K_out_of_N(N=3, K=3, discount=discount)
# push!(envs, k_model3)
# push!(envargs, (name="K-out-of-N (3)",))

### CustomGridWorlds
include("Environments/CustomGridworld.jl"); using .CustomGridWorlds
# # # Frozen Lake variants

# lakesmall = FrozenLakeSmall
# lakesmall.discount = discount
# push!(envs, lakesmall)
# push!(envargs, (name="Frozen Lake (4x4)",))

# lakelarge = FrozenLakeLarge
# lakelarge.discount = discount
# push!(envs, lakelarge)
# push!(envargs, (name="Frozen Lake (10x10)",))

### Hallway Envs

# minihallway = CustomMiniHallway
# minihallway.discount = discount
# push!(envs, minihallway)
# push!(envargs, (name="MiniHallway", ))

# hallway1 = Hallway1
# hallway1.discount = discount
# push!(envs, hallway1)
# push!(envargs, (name="Hallway1",))

# hallway2 = Hallway2
# hallway2.discount = discount
# push!(envs, hallway2)
# push!(envargs, (name="Hallway2",))

# tigergrid = TigerGrid
# tigergrid.discount = discount
# push!(envs, tigergrid)
# push!(envargs, (name="TigerGrid",))

# ### Explicit Spares Hallways (Wietze)
include("Environments/Sparse_models/SparseModels.jl"); using .SparseModels

# hallway1 = SparseHallway1(discount=discount)
# push!(envs, hallway1)
# push!(envargs, (name="Hallway1 (Sparse)",))

# hallway2 = SparseHallway2(discount=discount)
# push!(envs, hallway2)
# push!(envargs, (name="Hallway2 (Sparse)",))

# tigergrid = SparseTigerGrid(discount=discount)
# push!(envs, tigergrid)
# push!(envargs, (name="TigerGrid (Sparse)",))

# # ### DroneSurveilance
# # import DroneSurveillance
# # dronesurv = DroneSurveillance.DroneSurveillancePOMDP()
# # push!(envs, dronesurv)
# # push!(envargs, (name="DroneSurveilance",))

 ### Tag
#  using TagPOMDPProblem
#  tag = TagPOMDPProblem.TagPOMDP(discount_factor=discount)
#  push!(envs, tag)
#  push!(envargs, (name="Tag",))

# ### Mini Hallway
# minihall = POMDPModels.MiniHallway()
# push!(envs, minihall)
# push!(envargs, (name="MiniHallway",))

# ### TMaze (Does not work with FIB)
# tmaze = POMDPModels.TMaze()
# POMDPs.reward(tmaze::TMaze, s::POMDPTools.ModelTools.TerminalState,a ) = 0
# push!(envs, tmaze)
# push!(envargs, (name="TMaze",))

# For some reason, the envs below do not work:

### SubHunt (No 'observations' defined)
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
#                           Run Solvers 
##################################################################

verbose = true
sims, steps = 100, 100

policy_names = map(sarg -> sarg.name, solverargs)
env_names = map(envarg -> envarg.name, envargs)
nr_pols, nr_envs = length(policy_names), length(env_names)

# upperbounds_init = zeros(nr_envs, nr_pols)
# upperbounds_sampled = zeros( nr_envs, nr_pols)
# return_means = zeros( nr_envs, nr_pols)
# time_solve = zeros( nr_envs, nr_pols)
# time_online = zeros( nr_envs, nr_pols)

models_to_skip_WBIB = ["RockSample (10x10)","Frozen Lake (10x10)","2-out-of-2","3-out-of-3", "Tag"]

# t = time()
# open("run_out_$t.txt", "w") do file
#     write(file, "Policies:  ")
#     for p in policy_names
#         write(file, "$p \t")
#     end
#     write(file, "\n")

for (m_idx,(model, modelargs)) in enumerate(zip(envs, envargs))
    # model = SparseTabularPOMDP(model)
    verbose && println("Testing in $(modelargs.name) environment")
    policies = []
    upperbounds_init = zeros(nr_pols)
    upperbounds_sampled = zeros(nr_pols)
    return_means = zeros(nr_pols)
    time_solve = zeros(nr_pols)
    time_online = zeros(nr_pols)
    
    # Calculate & print model size
    constants = BIB.get_constants(model)
    SAO_probs, SAOs = BIB.get_all_obs_probs(model; constants)
    B, B_idx = BIB.get_belief_set(model, SAOs; constants)
    Data = BIB.BIB_Data(zeros(2,2), B, B_idx, SAO_probs, SAOs, Dict(zip(constants.S, 1:constants.ns)), constants)
    BBao_data = BIB.get_Bbao(model, Data, constants)
    ns, na, no, nb = constants.ns, constants.na, constants.no, length(B)
    nbao = length(BBao_data.Bbao) + length(B)
    verbose && println("|S| = $ns, |A| = $na, |O| = $no, |B| = $nb, |BAO| = $nbao")


    for (s_idx,(solver, solverarg)) in enumerate(zip(solvers, solverargs))
        if solverarg.name == "BIBSolver (worst-case)" && modelargs.name in models_to_skip_WBIB         
            continue
        end 

        verbose && println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)

        # Compute policy & get upper bound
        t = @elapsed begin
            policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...) 
        end
        @profile (policy, info = solve_info(solver, model; solverarg.pargs...))
        pprof(;webport=58699)
        (info isa Nothing) ? val = POMDPs.value(policy, POMDPs.initialstate(model)) : val = info.value        
        verbose && println("Upperbound $val (computed in $t seconds)")
        upperbounds_init[s_idx] = val
        push!(policies, policy)
        time_solve[s_idx] = t

        # Simulate policy & get avg returns
        # verbose && print("Simulating policy...")
        # rs = []
        # for i=1:sims
        #     rtot = 0
        #     for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
        #         rtot += POMDPs.discount(model)^(t-1) * r
        #     end
        #     push!(rs,rtot)
        # end
        # rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
        # verbose && println("Returns: mean = $rs_avg, min = $rs_min, max = $rs_max")
        # return_means[s_idx] = rs_avg

    end

    # Simulate upper bounds on different sampled beliefs & get avg differences
    # if !(model isa RockSample.RockSamplePOMDP)
    #     upperbounds_sampled[:], time_online[:] = sample_avg_accuracy(model, policies)
    # end
    verbose && println("...")

    # data = [upperbounds_init, time_solve, time_online, return_means, upperbounds_sampled]
    # data = [upperbounds_init, time_solve, return_means]
    # names = ["bound:\t\t", "solve time:\t", "comp time:\t", "retuns:\t\t", "avg bound:\t" ]
    # for (d_idx, d) in enumerate(data) 
    #     write(file, names[d_idx])
    #     for p_idx in 1:length(policy_names)
    #         write(file, "$(round(d[p_idx],sigdigits=3))\t")
    #     end
    #     write(file, "\n")
    # end


# end
end