import POMDPs, POMDPTools
using POMDPs: simulate
using POMDPTools: stepthrough
include("BIB.jl")
using .BIB
using Statistics, POMDPModels

##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs = [], []

### BIB
push!(solvers, SBIBSolver)
push!(solverargs, (name="BIBSolver (standard)", sargs=(max_iterations=25, precision=1e-2), pargs=(), get_Q0=true))

### WBIB
push!(solvers, WBIBSolver)
push!(solverargs, (name="BIBSolver (worst-case)", sargs=(max_iterations=5, precision=1e-2), pargs=(), get_Q0=true))

### FIB
using FIB
push!(solvers, FIB.FIBSolver)
push!(solverargs, (name="FIB", sargs=(), pargs=(), get_Q0=false))

# ### SARSOP (Native)
# using NativeSARSOP
# push!(solvers, NativeSARSOP.SARSOPSolver)
# push!(solverargs, (name="Native SARSOP", sargs=(epsilon=0.5, precision=-10.0, kappa=0.5, delta=1e-1, max_time=5.0, verbose=true), pargs=(), get_Q0=false))

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

### RockSample
import RockSample
map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
rocksample = RockSample.RockSamplePOMDP(map_size, rock_pos)
push!(envs, rocksample)
push!(envargs, (name="RockSample",))

# push!(envargs, (name="LaserTag",))

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
# reward(tmaze::Any, s::POMDPTools.ModelTools.TerminalState,a ) = 0
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

##################################################################
#                           Run Solvers 
##################################################################

sims, steps = 250, 50

for (model, modelargs) in zip(envs, envargs)
    println("Testing in $(modelargs.name) environment")
    for (solver, solverarg) in zip(solvers, solverargs)
        println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)
        # simulator = POMDPTools.RolloutSimulator(max_steps=steps)
        # simulator = StepSimulator(max_steps=steps)

        @time policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...)
        solverarg.get_Q0 && println("Value for b: ", bvalue(policy, POMDPs.initialstate(model)))

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
        #TODO: export 
    end
    println("########################")
end