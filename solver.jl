#########################################
#               Solver:
#########################################

abstract type BIBSolver <: Solver end

struct SBIBSolver <: BIBSolver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end
SBIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3) = SBIBSolver(max_iterations, precision)

struct WBIBSolver <: BIBSolver
    max_iterations::Int64
    precision::Float64
 end
WBIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3) = WBIBSolver(max_iterations, precision)

struct BIB_Data
    Q::Array{Float64,2}
    B::Vector
    B_idx::Array{Int,3}
    SAO_probs::Array{Float64,3}
    SAOs::Array{Vector{Int},2}
end

struct BBAO_Data
    Bbao::Vector
    Bbao_idx::Array{Tuple{Bool,Int},3}
    B_overlap::Array{Vector{Int}}
    Bbao_overlap::Array{Vector{Int}}
end
function get_bao(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int, B)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai,oi]
    in_B ? (return B[baoi]) : (return Bbao_data.Bbao[baoi])
end
function get_overlap(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai,oi]
    in_B ? (return Bbao_data.B_overlap[baoi]) : (return Bbao_data.Bbao_overlap[baoi])
end

struct C
    S
    A 
    O 
    ns 
    na
    no
end

function POMDPs.solve(solver::X, model::POMDP) where X<:BIBSolver

    S, A, O = states(model), actions(model), observations(model) #TODO: figure out if I need to use ordered_... here
    ns, na, no = length(states(model)), length(actions(model)), length(observations(model))
    constants = C(S,A,O,ns,na,no)

    # 1: Precompute observation probabilities
    SAO_probs = []                                  # ∀s,a,o, gives probability of o given (s,a)
    SAOs = []                                       # ∀s,a, gives os with p>0
    SAO_probs, SAOs = get_all_obs_probs(model; constants)

    # 2 : Pre-compute all beliefs after 1 step
    B = []                                          # Our set of beliefs
    B_idx = []                                      # ∀s,a,o, gives index of b_sao in B
    B, B_idx = get_belief_set(model, SAOs; constants)

    # 3 : If WBIB or EBIB, precompute all beliefs after 2 steps
    Bbao_data = []
    if solver isa WBIBSolver
        Bbao_data = get_Bbao(model, B, constants)
        # printdb(Bbao)
        # printdb(Bbao_idx)
        # printdb(B_overlap)
    end

    # 4 : Compute Q-values bsoa beliefs
    Qs = get_QMDP_Beliefset(model, B, constants)    # ∀b∈B, contains QBIB value (initialized using QMDP)

    # Lets be overly fancy! Define a function for computing Q, depending on the specific solver
    get_Q, args = identity, []
    if solver isa SBIBSolver
        pol, get_Q, args = SBIBPolicy, get_QBIB_Beliefset, (model, Qs, B, B_idx, SAOs, SAO_probs, constants)
    elseif solver isa WBIBSolver
        pol, get_Q, args = WBIBPolicy, get_QWBIB_Beliefset, (model, Qs, B, Bbao_data, SAO_probs, constants)
    else
        throw("Solver type not recognized!")
    end
    
    # Now iterate:
    for i=1:solver.max_iterations
        printdb(i)
        Qs, max_dif = get_Q(args...)
        max_dif < solver.precision && (printdb("breaking after $i iterations:"); break)
    end
    return pol(model, BIB_Data(Qs,B,B_idx,SAO_probs,SAOs) )
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

"""
For each belief-action-observation pair, compute which beliefs from our set have non-zero overlap.
"""
function get_Bbao(model, B, constants)
    U = DiscreteHashedBeliefUpdater(model)
    nb = length(B)

    Bbao = []
    Bbao_idx = Array{Tuple{Bool,Int}}(undef, nb, constants.na, constants.no)

    # Record bao: reference B if it's already in there, otherwise add to Bbao
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (oi,o) in enumerate(constants.O)
                bao = update(U,b,a,o)
                in_B = true
                k = findfirst( x-> x==bao, B)
                if isnothing(k)
                    in_B = false
                    k = findfirst(x -> x==bao, Bbao)
                    isnothing(k) && (push!(Bbao, bao); k=length(Bbao))
                end
                Bbao_idx[bi,ai,oi] = (in_B, k)                
            end
        end
    end
    nbao = length(Bbao)
    B_overlap = Array{Vector{Int}}(undef, nb)
    Bbao_overlap = Array{Vector{Int}}(undef, nbao)

    # Record overlap for b
    for (bi,b) in enumerate(B)
        B_overlap[bi] = []
        for (bpi,bp) in enumerate(B)
            for s in support(b)
                if pdf(bp,s) > 0
                    push!(B_overlap[bi], bpi)
                    break
                end
            end
        end
    end
    # Record overlap for bao
    for (bi,b) in enumerate(Bbao)
        Bbao_overlap[bi] = []
        for (bpi,bp) in enumerate(B)
            for s in support(b)
                if pdf(bp,s) > 0
                    push!(Bbao_overlap[bi], bpi)
                    break
                end
            end
        end
    end


    return BBAO_Data(Bbao, Bbao_idx, B_overlap, Bbao_overlap)
end


#########################################
#            Value Computations:
#########################################

############ QMDP ###################

function get_QMDP_Beliefset(model::POMDP, B::Vector, constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")

    π_QMDP = solve(QMDPSolver_alt(), model)
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (si, s) in enumerate(constants.S)
                if pdf(b,s) > 0
                    Qs[b_idx, ai] += pdf(b,s) * π_QMDP.Q_MDP[si,ai]
                end
            end
        end
    end
    return Qs
end

############ BIB ###################

function get_QBIB_Beliefset(model::POMDP,Qs,B::Vector,B_idx, SAOs, SAO_probs, constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")

    Qs_new = zero(Qs) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
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


############ WBIB ###################

function get_QWBIB_Beliefset(model::POMDP,Qs,B::Vector, Bbao_data::BBAO_Data, SAO_probs, constants::Union{C,Nothing}=nothing)
    Qs_new = zero(Qs) # TODO: this may be inefficient?
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Qs_new[bi,ai] = get_QWBIB_ba(model,b,a, Qs, B, SAO_probs; ai=ai, Bbao_data=Bbao_data, bi=bi )
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Qs)) ./ (Qs.+1e-10))
    return Qs_new, max_dif
end

get_QWBIB_ba(model::POMDP, b,a,D::BIB_Data; ai=nothing) = get_QWBIB_ba(model, b, a, D.Qs, D.B, D.SAO_probs; ai)
function get_QWBIB_ba(model::POMDP,b,a,Qs,B, SAO_probs; ai=nothing, Bbao_data=nothing, bi=nothing)
    #TODO: filter B on beliefs that have overlap in support, ignore Returns
    S_dict = Dict( zip(states(model), 1:length(states(model))) )
    Q = breward(model,b,a)
    for (oi, o) in enumerate(observations(model))
        Qo = -Inf
        for (api, ap) in enumerate(actions(model)) #TODO: would it be quicker to let the solver also find the best action?
            thisQo = 0
            if !(Bbao_data isa Nothing) && !(bi isa Nothing)
                bao = get_bao(Bbao_data, bi, ai, oi, B)
                overlap_idxs = get_overlap(Bbao_data, bi, ai, oi)
                thisBs, thisQs = B[overlap_idxs], Qs[overlap_idxs, api]
                thisQo = get_QLP(bao, thisQs, thisBs)
            else
                bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
                thisQo = get_QLP(bao, Qs[:,api],B)
            end
            Qo = max(Qo, thisQo)
        end
        p = 0
        for s in support(b)
            p += pdf(b,s) * SAO_probs[oi,S_dict[s],ai]
        end
        Q += p * discount(model) * Qo
    end
    return Q
end

""" Uses a point-set B with value estimates Qs to estimate the value of a belief b."""
function get_QLP(b,Qs,B)
    # printdb(b)
    # printdb(B)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, b_ps[1:length(B)] in Semicontinuous(0.0,1.0))
    for s in support(b)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B)
            p = pdf(bp,s)
            if p > 0
                push!(Idx, bpi)
                push!(Ps,p)
            end
        end
        @constraint(model, sum(b_ps[Idx[i]] * Ps[i] for i in 1:length(Idx)) == pdf(b,s) )
    end
    @objective(model, Min, sum(Qs .* b_ps))
    # printdb(model)
    optimize!(model)
    return(objective_value(model))
end


#########################################
#               Policy:
#########################################

abstract type BIBPolicy <: Policy end
POMDPs.updater(π::X) where X<:BIBPolicy = DiscreteHashedBeliefUpdater(π.model)

struct SBIBPolicy <: BIBPolicy
    model::POMDP
    Data::BIB_Data
end

struct WBIBPolicy <: BIBPolicy
    model::POMDP
    Data::BIB_Data
end

POMDPs.action(π::X, b) where X<: BIBPolicy = first(action_value(π, b))
bvalue(π::X, b) where X<: BIBPolicy = last(action_value(π,b))

function action_value(π::SBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::WBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end





