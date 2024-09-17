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

    include("Beliefs.jl")
    include("Caching.jl")
    # Both QMDP and FIB use alpha-vectors in the original version, which is slow...
    include("SimpleHeuristics.jl")
    include("solver.jl")
    include("Convenience.jl")

    export
    
    # Convenience:

    # Beliefs:
    DiscreteHashedBelief, DiscreteHashedBeliefUpdater, 

    #Solver:
    # BIBPolicy,
    SBIBSolver, WBIBSolver, EBIBSolver, 
    SBIBPolicy, WBIBPolicy, EBIBPolicy, action_value,

    # QS_table_policy,
    QMDPSolver_alt, FIBSolver_alt,
    QMDPPlanner_alt, FIBPlanner_alt,

    get_heuristic_pointset
    
end