struct WBIBSolver <: Solver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end

function WBIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3)
    return WBIBSolver(max_iterations, precision)
end

struct BIB_Data
    Q::Array{Float64,2}
    B_idx::Array{Int,3}
    SAO_probs::Array{Float64,3}
    SAOs::Array{Vector{Int},2}
end

struct C
    S
    A 
    O 
    ns 
    na
    no
end

function POMDPs.solve(solver::BIBSolver, model::POMDP)

    S, A, O = states(model), actions(model), observations(model) #TODO: figure out if I need to use ordered_... here
    ns, na, no = length(states(model)), length(actions(model)), length(observations(model))
    constants = C(S,A,O,ns,na,no)

    # 1: Precompute observation probabilities
    SAO_probs = []                                  # ∀s,a,o, gives probability of o given (s,a)
    SAOs = []                                       # ∀s,a, gives os with p>0
    SAO_probs, SAOs = get_all_obs_probs(model; constants)

    # 2 : Pre-compute all beliefs
    B = []                                          # Our set of beliefs
    B_idx = []                                      # ∀s,a,o, gives index of b_sao in B
    B, B_idx = get_belief_set(model, SAOs; constants)

    # 3 : Compute Q-values bsoa beliefs
    Qs = get_QMDP_Beliefset(model, B; constants)    # ∀b∈B, contains QBIB value (initialized using QMDP)
    for i=1:solver.max_iterations
        Qs, max_dif = get_QBIB_Beliefset(model,Qs,B,B_idx,SAOs, SAO_probs; constants)
        max_dif < solver.precision && (printdb("breaking after $i iterations:"); break)
    end
    return BIBPolicy(model, BIB_Data(Qs,B_idx,SAO_probs,SAOs) )
end


