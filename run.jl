import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles
include("BIB.jl")
using .BIB
using Statistics, POMDPModels
import RockSample

##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs = [], []

iters, tol = 250, 1e-5

### FIB
using FIB
push!(solvers, FIB.FIBSolver)
push!(solverargs, (name="FIB", sargs=(max_iterations=iters,tolerance=tol), pargs=(), get_Q0=true))

### BIB
push!(solvers, SBIBSolver)
push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### EBIB
# push!(solvers, EBIBSolver)
# push!(solverargs, (name="BIBSolver (entropy)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### WBIBs
# push!(solvers, WBIBSolver)
# push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=250, precision=1e-5), pargs=(), get_Q0=true))

# SARSOP
include("Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt
max_time = 30.0

push!(solvers, NativeSARSOP_alt.SARSOPSolver)
push!(solverargs, (name="SARSOP (max $max_time s)", sargs=( precision=1e-5, max_time=max_time, max_steps=1, verbose=false), pargs=(), get_Q0=true))

# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# solver = EBIBSolver(max_iterations=250, precision=1e-5)
# push!(solverargs, (name="SARSOP+BIB (max $max_time s)", sargs=(epsilon=0.5, precision=1e-5, kappa=0.5, delta=1e-1, max_time=max_time, max_steps=1, verbose=false, heuristic_solver=solver), pargs=(), get_Q0=true))

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

# ### ABC
# include("Environments/ABCModel.jl"); using .ABCModel
# abcmodel = ABC()
# discount(::ABC) = 0.99
# push!(envs, abcmodel)
# push!(envargs, (name="ABCModel",))

# ### Tiger
# tiger = POMDPModels.TigerPOMDP()
# tiger.discount_factor = 0.99
# push!(envs, tiger)
# push!(envargs, (name="Tiger",))

# ### RockSample
# import RockSample
# # This env is very difficult to work with for some reason...
# POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
# POMDPs.discount(M::RockSample.RockSamplePOMDP) = 0.99

# map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
# rocksamplesmall = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envargs, (name="RockSample (5x5)",))
# push!(envs, rocksamplesmall)

# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
# rocksamplelarge = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envargs, (name="RockSample (10x10)",))
# push!(envs, rocksamplelarge)

# # ### K-out-of-N
# include("Environments/K-out-of-N.jl"); using .K_out_of_Ns

# k_model2 = K_out_of_N(2, 2)
# push!(envs, k_model2)
# push!(envargs, (name="K-out-of-N (2)",))

# k_model3 = K_out_of_N(3, 3)
# push!(envs, k_model3)
# push!(envargs, (name="K-out-of-N (3)",))

# # Frozen Lake esque
# include("Environments/GridWorldPOMDP.jl"); using .AMGridworlds

# lakesmall = FrozenLakeSmall
# push!(envs, lakesmall)
# push!(envargs, (name="Frozen Lake (4x4)",))

# lakelarge = FrozenLakeLarge
# push!(envs, lakelarge)
# push!(envargs, (name="Frozen Lake (10x10)",))

### CustomGridWorlds
include("Environments/CustomGridworld.jl"); using .CustomGridWorlds
# lakesmall = FrozenLakeSmall
# push!(envs, lakesmall)
# push!(envargs, (name="Frozen Lake (4x4)",))

# lakelarge = FrozenLakeLarge
# push!(envs, lakelarge)
# push!(envargs, (name="Frozen Lake (10x10)",))

minihallway = CustomMiniHallway
push!(envs, minihallway)
push!(envargs, (name="MiniHallway", ))

# hallway1 = Hallway1
# push!(envs, hallway1)
# push!(envargs, (name="Hallway1",))

# hallway2 = Hallway2
# push!(envs, hallway2)
# push!(envargs, (name="Hallway2",))

# hallway3 = Hallway3
# push!(envs, hallway3)
# push!(envargs, (name="Hallway3",))

# # ### DroneSurveilance
# # import DroneSurveillance
# # dronesurv = DroneSurveillance.DroneSurveillancePOMDP()
# # push!(envs, dronesurv)
# # push!(envargs, (name="DroneSurveilance",))

# ### Tag
# using TagPOMDPProblem
# discount(m::TagPOMDP) = 0.99
# tag = TagPOMDPProblem.TagPOMDP()
# push!(envs, tag)
# push!(envargs, (name="Tag",))

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
sims, steps = 10, 10

function sample_avg_accuracy(model::POMDP, policies::Vector; samples::Int=100, distance::Int=5, samplepolicy=nothing)
    if isnothing(samplepolicy)
        samplepolicy = POMDPTools.RandomPolicy(model; updater=DiscreteHashedBeliefUpdater(model))
    end
    relative_bounds = zeros(length(policies))

    cum_times = zeros(length(policies))
    bs = []
    for (i,data) in enumerate(stepthrough(model,samplepolicy, "b,s,a,o,r"; max_steps=(samples+1)*distance))
        b = data.b
        i % distance == 0 && push!(bs,b)
    end
    for b in bs
        bounds = []
        for (i,policy) in enumerate(policies)
            t = @elapsed begin bound = POMDPs.value(policy, b) end
            push!(bounds, bound)
            cum_times[i] += t
        end
        best_bound = minimum(bounds)
        bounds =  (bounds .- best_bound) ./ abs(best_bound)
        relative_bounds = relative_bounds .+ bounds
    end
    return (relative_bounds ./ samples), (cum_times ./ samples)
end

policy_names = map(sarg -> sarg.name, solverargs)
env_names = map(envarg -> envarg.name, envargs)
nr_pols, nr_envs = length(policy_names), length(env_names)

upperbounds_init = zeros(nr_envs, nr_pols)
upperbounds_sampled = zeros( nr_envs, nr_pols)
return_means = zeros( nr_envs, nr_pols)
time_solve = zeros( nr_envs, nr_pols)
time_online = zeros( nr_envs, nr_pols)

models_to_skip_WBIB = ["RockSample (10x10)","Frozen Lake (10x10)","2-out-of-2","3-out-of-3", "Tag"]

t = time()
open("run_out_$t.txt", "w") do file
    write(file, "Policies:  ")
    for p in policy_names
        write(file, "$p \t")
    end
    write(file, "\n")

    for (m_idx,(model, modelargs)) in enumerate(zip(envs, envargs))
        verbose && println("Testing in $(modelargs.name) environment")
        write(file, "\n\n$(modelargs.name): \n")
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
        ns, na, no, nb = constants.ns, constants.na, constants.no, length(B)
        nbao = nb * na * no
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
            (info isa Nothing) ? val = POMDPs.value(policy, POMDPs.initialstate(model)) : val = info.value        
            verbose && println("Upperbound $val (computed in $t seconds)")
            upperbounds_init[s_idx] = val
            push!(policies, policy)
            time_solve[s_idx] = t

            # Simulate policy & get avg returns
            verbose && print("Simulating policy...")
            rs = []
            for i=1:sims
                rtot = 0
                for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
                    rtot += POMDPs.discount(model)^(t-1) * r
                end
                push!(rs,rtot)
            end
            rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
            verbose && println("Returns: mean = $rs_avg, min = $rs_min, max = $rs_max")
            return_means[s_idx] = rs_avg

        end

        # Simulate upper bounds on different sampled beliefs & get avg differences
        # if !(model isa RockSample.RockSamplePOMDP)
        #     upperbounds_sampled[:], time_online[:] = sample_avg_accuracy(model, policies)
        # end
        verbose && println("...")

        data = [upperbounds_init, time_solve, time_online, return_means, upperbounds_sampled]
        data = [upperbounds_init, time_solve, return_means]
        names = ["bound:\t\t", "solve time:\t", "comp time:\t", "retuns:\t\t", "avg bound:\t" ]
        for (d_idx, d) in enumerate(data) 
            write(file, names[d_idx])
            for p_idx in 1:length(policy_names)
                write(file, "$(round(d[p_idx],sigdigits=3))\t")
            end
            write(file, "\n")
        end


    end
end






