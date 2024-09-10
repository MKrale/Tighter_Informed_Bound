module BIB
    using POMDPs, POMDPTools, Random, Distributions, SparseArrays, Optimization, JuMP, Gurobi, HiGHS, Tulip, Memoization, LRUCache

    # Surpressing Gurobis printing...
    oldstd = stdout
    redirect_stdout(devnull)
    const GRB_ENV=Gurobi.Env()
    redirect_stdout(oldstd)

    
    printdb(x) = print(x,"\n")
    function printdb(x,y...)
        print(x,", ")
        printdb(y...)
    end

    include("QMDP_alt.jl") # This was easier than using the POMDP standard QMDP, which uses alpha-vectors for some reason...
    include("Beliefs.jl")
    include("solver.jl")
    include("Convenience.jl")

    export
    
    # Convenience:
    get_pointset_Sarsop,

    # Beliefs:
    DiscreteHashedBelief, DiscreteHashedBeliefUpdater, 

    #Solver:
    SBIBSolver, WBIBSolver, EBIBSolver, 
    SBIBPolicy, WBIBPolicy, EBIBPolicy, action_value 
    
end