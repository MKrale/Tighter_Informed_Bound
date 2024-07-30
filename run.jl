import POMDPs, POMDPTools
using POMDPs
using POMDPTools: stepthrough
include("BIB.jl")
using .BIB
using Statistics, POMDPModels

##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs = [], []

# ### BIB
# push!(solvers, SBIBSolver)
# push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=20, precision=1e-3), pargs=(), get_Q0=true))

# ### EBIB
# push!(solvers, EBIBSolver)
# push!(solverargs, (name="BIBSolver (entropy)", sargs=(max_iterations=20, precision=1e-3), pargs=(), get_Q0=true))

# ### WBIB
# push!(solvers, WBIBSolver)
# push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=10, precision=1e-3), pargs=(), get_Q0=true))

# ### FIB
# using FIB
# push!(solvers, FIB.FIBSolver)
# push!(solverargs, (name="FIB", sargs=(), pargs=(), get_Q0=true))

### SARSOP + BIB
include("Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt
push!(solvers, NativeSARSOP_alt.SARSOPSolver)
push!(solverargs, (name="SARSOP+BIB", sargs=(epsilon=0.5, precision=-10.0, kappa=0.5, delta=1e-1, max_time=5.0, verbose=true), pargs=(), get_Q0=true))

### SARSOP
using NativeSARSOP
push!(solvers, NativeSARSOP.SARSOPSolver)
push!(solverargs, (name="SARSOP", sargs=(epsilon=0.5, precision=-10.0, kappa=0.5, delta=1e-1, max_time=5.0, verbose=true), pargs=(), get_Q0=true))

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
# push!(envs, abcmodel)
# push!(envargs, (name="ABCModel",))

# ### Tiger
# tiger = POMDPModels.TigerPOMDP()
# tiger.discount_factor = 0.9
# push!(envs, tiger)
# push!(envargs, (name="Tiger",))

# ### RockSample
# import RockSample
# POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
# # map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
# # push!(envargs, (name="RockSample (default)",))
# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
# push!(envargs, (name="RockSample (10x10)",))
# rocksample = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envs, rocksample)

### K-out-of-N
include("Environments/K-out-of-N.jl"); using .K_out_of_Ns
N, K = 3, 3
k_model = K_out_of_N(N, K)
push!(envs, k_model)
push!(envargs, (name="$K-out-of-$N",))

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

sims, steps = 50, 50


for (model, modelargs) in zip(envs, envargs)
    println("Testing in $(modelargs.name) environment")
    
    constants = BIB.get_constants(model)
    SAO_probs, SAOs = BIB.get_all_obs_probs(model; constants)
    B, B_idx = BIB.get_belief_set(model, SAOs; constants)
    ns, na, no, nb = constants.ns, constants.na, constants.no, length(B)
    nbao = nb * na * no
    println("|S| = $ns, |A| = $na, |O| = $no, |B| = $nb, |BAO| = $nbao")
    for (solver, solverarg) in zip(solvers, solverargs)
        println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)
        # simulator = POMDPTools.RolloutSimulator(max_steps=steps)
        # simulator = StepSimulator(max_steps=steps)

        @time policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...)
        solverarg.get_Q0 && println("Value for b: ", POMDPs.value(policy, POMDPs.initialstate(model)))

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
        # #TODO: export 
    end
    println("########################")
end