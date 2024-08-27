module BIB
    using POMDPs, POMDPTools, Random, Distributions, SparseArrays, JuMP, Gurobi, HiGHS, Memoization, LRUCache
    const GRB_ENV=Gurobi.Env()
    
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