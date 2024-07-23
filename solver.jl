#########################################
#               Solver:
#########################################

struct BIBSolver <: Solver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end

function BIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3)
    return BIBSolver(max_iterations, precision)
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

#########################################
#            Precomputation:
#########################################

# Observation Probabilities:

"""
Computes the probabilities of each observation for each state-action pair. \n 
Returns: SAO_probs ((s,a,o)->p), SAOs ((s,a) -> os with p>0)
"""
function get_all_obs_probs(model::POMDP; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S,A,O = constants.S , constants.A, constants.O
    ns, na, no = constants.ns, constants.na, constants.no

    SAO_probs = zeros(no,ns,na)
    SAOs = Array{Vector{Int}}(undef,ns,na)
    # TODO: this can be done cleaner I imagine...
    for si in 1:ns
        for sa in 1:na
            SAOs[si,sa] = []
        end
    end

    for (oi,o) in enumerate(O)
        for (si,s) in enumerate(S)
            for (ai,a) in enumerate(A)
                SAO_probs[oi,si,ai] = get_obs_prob(model,o,s,a)
                SAO_probs[oi,si,ai] > 0 && push!(SAOs[si,ai], oi)
            end
        end
    end
    return SAO_probs, SAOs
end

"""Computes the probability of an observation given a state-action pair."""
function get_obs_prob(model::POMDP, o,s,a)
    p = 0
    for (sp, psp) in weighted_iterator(transition(model,s,a))
        dist = observation(model,a,sp)
        p += psp*pdf(dist,o)
    end
    return p
end
"""Computes the probability of an observation given a belief-action pair."""
get_obs_prob(model::POMDP, o, b::DiscreteHashedBelief, a) = sum( (s,p) -> p*get_obs_prob(model,o,s,a), zip(b.state_list, b.probs) )

# Belief Sets:

"""
Computes all beliefs reachable in one step from a state. \n
Returns: B (vector of beliefs), B_idx (s,a,o -> index of B) 
"""
function get_belief_set(model, SAOs; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no
    U = DiscreteHashedBeliefUpdater(model)

    B = Array{DiscreteHashedBelief,1}()       
    B_idx = zeros(Int,ns,na,no)
    push!(B, DiscreteHashedBelief(initialstate(model)))
    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief([s],[1.0])
        for (ai,a) in enumerate(A)
            for oi in SAOs[si,ai]
                o = O[oi]
                b = update(U, b_s, a, o)
                k = findfirst( x -> x==b , B)
                isnothing(k) && (push!(B,b); k=length(B))
                B_idx[si,ai,oi] = k
            end
        end
    end
    return B, B_idx
end

#########################################
#            Value Computations:
#########################################

function get_QMDP_Beliefset(model::POMDP, B::Vector; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no

    π_QMDP = solve(QMDPSolver_alt(), model)
    Qs = zeros(Float64, length(B), na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(A)
            for (si, s) in enumerate(S)
                if pdf(b,s) > 0
                    Qs[b_idx, ai] += pdf(b,s) * π_QMDP.Q_MDP[si,ai]
                end
            end
        end
    end
    return Qs
end

function get_QBIB_Beliefset(model::POMDP,Qs,B::Vector,B_idx, SAOs, SAO_probs; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no

    Qs_new = zero(Qs) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(B)
        for (ai, a) in enumerate(A)
            Qs_new[b_idx,ai] = get_QBIB_ba(model,b,a, Qs, B_idx, SAO_probs, SAOs ; ai=ai)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Qs)) ./ (Qs.+1e-10))
    return Qs_new, max_dif
end

function get_QBIB_ba(model::POMDP,b,a,Qs,B_idx,SAO_probs, SAOs; ai=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), actions(model)) )
    S_dict = Dict( zip(states(model), 1:length(states(model))) )
    Q = breward(model,b,a)
    for (oi, o) in enumerate(observations(model))
        Qo = -Inf
        for (api, ap) in enumerate(actions(model))
            Qoa = 0
            for s in support(b)
                si = S_dict[s]
                if oi in SAOs[si,ai]
                    p_obs = SAO_probs[oi,si,ai]
                    p = pdf(b,s) * p_obs
                    bp_idx = B_idx[si,ai,oi]
                    Qoa += p * Qs[bp_idx,api]
                end
            end
            Qo = max(Qo, Qoa)
            
        end
        Q += discount(model) * Qo
    end
    return Q
end
get_QBIB_ba(model::POMDP,b,a,D::BIB_Data; ai=nothing) = get_QBIB_ba(model,b,a,D.Q, D.B_idx, D.SAO_probs, D.SAOs; ai)

#########################################
#               Policy:
#########################################

struct BIBPolicy <:Policy
    model::POMDP
    Data::BIB_Data
end
POMDPs.updater(π::BIBPolicy) = DiscreteHashedBeliefUpdater(π.model)

function action_value(π::BIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

POMDPs.action(π::BIBPolicy, b) = first(action_value(π, b))
bvalue(π::BIBPolicy, b) = last(action_value(π,b))