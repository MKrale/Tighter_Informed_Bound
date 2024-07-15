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
push!(solvers, BIBSolver)
push!(solverargs, (name="BIBSolver", sargs=(max_iterations=10, precision=1e-2), pargs=()))

### FIB
using FIB
push!(solvers, FIB.FIBSolver)
push!(solverargs, (name="FIB", sargs=(), pargs=()))

# SARSOP (Native)
using NativeSARSOP
push!(solvers, NativeSARSOP.SARSOPSolver)
push!(solverargs, (name="Native SARSOP", sargs=(epsilon=0.5, precision=-10.0, kappa=0.5, delta=1e-1, max_time=5.0, verbose=true), pargs=()))

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



##################################################################
#                           Run Solvers 
##################################################################

sims, steps = 10, 50

for (model, modelargs) in zip(envs, envargs)
    println("Testing in $(modelargs.name) environment")
    for (solver, solverarg) in zip(solvers, solverargs)
        println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)
        # simulator = POMDPTools.RolloutSimulator(max_steps=steps)
        # simulator = StepSimulator(max_steps=steps)

        @time policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...)
        println("Done!")

        print("Simulating policy...")
        rs = []
        @time begin
            for i=1:sims
                rtot = 0
                for (b,s,a,o,r) in stepthrough(model,policy,"b,s,a,o,r";max_steps=steps)
                    rtot += r
                end
                push!(rs,rtot)
            end
        end
        rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
        println("Done!\n(Returns: mean = $rs_avg, min = $rs_min, max = $rs_max)")
        #TODO: export 
    end
    println("########################")
end