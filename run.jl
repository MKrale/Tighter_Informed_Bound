import POMDPs, POMDPTools
using POMDPs
using POMDPTools
include("BIB.jl")
using .BIB
using Statistics, POMDPModels

##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs = [], []

# ### FIB
# using FIB
# push!(solvers, FIB.FIBSolver)
# push!(solverargs, (name="FIB", sargs=(), pargs=(), get_Q0=true))

# ### BIB
# push!(solvers, SBIBSolver)
# push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=100, precision=1e-5), pargs=(), get_Q0=true))

# ### EBIB
# push!(solvers, EBIBSolver)
# push!(solverargs, (name="BIBSolver (entropy)", sargs=(max_iterations=100, precision=1e-5), pargs=(), get_Q0=true))

### WBIB
push!(solvers, WBIBSolver)
push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=25, precision=1e-5), pargs=(), get_Q0=true))

# ### SARSOP + BIB
# include("Sarsop_altered/NativeSARSOP.jl")
# import .NativeSARSOP_alt
# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# push!(solverargs, (name="SARSOP+BIB", sargs=(epsilon=0.5, precision=0.001, kappa=0.5, delta=1e-1, max_time=2.0, max_steps=1, verbose=true), pargs=(), get_Q0=true))

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

### ABC
include("Environments/ABCModel.jl"); using .ABCModel
abcmodel = ABC()
push!(envs, abcmodel)
push!(envargs, (name="ABCModel",))

# ### Tiger
# tiger = POMDPModels.TigerPOMDP()
# tiger.discount_factor = 0.9
# push!(envs, tiger)
# push!(envargs, (name="Tiger",))

# ### RockSample
# import RockSample
# # This env is very difficult to work with for some reason...
# POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
# # map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
# # push!(envargs, (name="RockSample (default)",))
# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
# push!(envargs, (name="RockSample (10x10)",))
# rocksample = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envs, rocksample)

# ### K-out-of-N
# include("Environments/K-out-of-N.jl"); using .K_out_of_Ns
# N, K = 2, 2
# k_model = K_out_of_N(N, K)
# push!(envs, k_model)
# push!(envargs, (name="$K-out-of-$N",))

# ### DroneSurveilance
# import DroneSurveillance
# dronesurv = DroneSurveillance.DroneSurveillancePOMDP()
# push!(envs, dronesurv)
# push!(envargs, (name="DroneSurveilance",))

# ### Tag
# import TagPOMDPProblem
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

sims, steps = 20, 10

function sample_avg_accuracy(model::POMDP, policies::Vector; samples::Int=100, distance::Int=5, samplepolicy=nothing)
    if isnothing(samplepolicy)
        samplepolicy = POMDPTools.RandomPolicy(model; updater=DiscreteHashedBeliefUpdater(model))
    end

    relative_bounds = zeros(length(policies))

    bs = []
    for (i,data) in enumerate(stepthrough(model,samplepolicy, "b,s,a,o,r"; max_steps=(samples+1)*distance))
        b = data.b
        i % distance == 0 && push!(bs,b)
    end
    for b in bs
        bounds = []
        for policy in policies
            push!(bounds, POMDPs.value(policy, b))
        end
        best_bound = minimum(bounds)
        bounds =  (bounds .- best_bound) ./ abs(best_bound)
        relative_bounds = relative_bounds .+ bounds
    end
    return relative_bounds ./ samples
end


for (model, modelargs) in zip(envs, envargs)
    println("Testing in $(modelargs.name) environment")
    
    constants = BIB.get_constants(model)
    SAO_probs, SAOs = BIB.get_all_obs_probs(model; constants)
    B, B_idx = BIB.get_belief_set(model, SAOs; constants)
    ns, na, no, nb = constants.ns, constants.na, constants.no, length(B)
    nbao = nb * na * no
    println("|S| = $ns, |A| = $na, |O| = $no, |B| = $nb, |BAO| = $nbao")
    policies = []
    for (solver, solverarg) in zip(solvers, solverargs)
        println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)
        # simulator = POMDPTools.RolloutSimulator(max_steps=steps)
        # simulator = StepSimulator(max_steps=steps)

        @time policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...)
        solverarg.get_Q0 && println("Value for b: ", POMDPs.value(policy, POMDPs.initialstate(model)))
        push!(policies, policy)

        print("Simulating policy...")
        rs = []
        @time begin
            for i=1:sims
                rtot = 0
                for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
                    rtot += POMDPs.discount(model)^(t-1) * r
                end
                push!(rs,rtot)
            end
        end
        rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
        println("Returns: mean = $rs_avg, min = $rs_min, max = $rs_max")

        # # #TODO: export 
    end

    println(sample_avg_accuracy(model, policies))
    println("########################")
end





