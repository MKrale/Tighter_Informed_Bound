#########################################
#               Solver:
#########################################

abstract type BIBSolver <: Solver end

struct SBIBSolver <: BIBSolver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end
SBIBSolver(;max_iterations::Int64=25, precision::Float64=1e-5) = SBIBSolver(max_iterations, precision)

struct WBIBSolver <: BIBSolver
    max_iterations::Int64
    precision::Float64
 end
WBIBSolver(;max_iterations::Int64=10, precision::Float64=1e-5) = WBIBSolver(max_iterations, precision)

struct EBIBSolver <: BIBSolver
    max_iterations::Int64
    precision::Float64
 end
EBIBSolver(;max_iterations::Int64=25, precision::Float64=1e-5) = EBIBSolver(max_iterations, precision)

struct C
    S; A; O 
    ns; na; no
end
get_constants(model) = C( states(model), actions(model), observations(model),
                         length(states(model)), length(actions(model)), length(observations(model)))

struct BIB_Data
    Q::Array{Float64,2}
    B::Vector
    B_idx::Array{Int,3}
    SAO_probs::Array{Float64,3}
    SAOs::Array{Vector{Int},2}
    S_dict::Dict{Any, Int}
    constants::C
end
BIB_Data(Q::Array{Float64,2}, D::BIB_Data) = BIB_Data(Q,D.B, D.B_idx, D.SAO_probs, D.SAOs, D.S_dict, D.constants)

struct BBAO_Data
    Bbao::Vector
    Bbao_idx::Array{Dict{Int,Tuple{Bool,Int}},2}
    B_overlap::Array{Vector{Int}}
    Bbao_overlap::Array{Vector{Int}}
    B_entropies::Array{Float64}
end
function get_bao(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int, B)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return B[baoi]) : (return Bbao_data.Bbao[baoi])
end
function get_overlap(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return Bbao_data.B_overlap[baoi]) : (return Bbao_data.Bbao_overlap[baoi])
end

struct Weights_Data
    B_weights::Vector{Vector{Tuple{Int,Float64}}}
    Bbao_weights::Vector{Vector{Tuple{Int,Float64}}}
end
function get_weights(Bbao_data, weights_data, bi, ai, oi)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return weights_data.B_weights[baoi]) : (return weights_data.Bbao_weights[baoi])
end
get_weights_indexfree(Bbao_data, weights_data,bi,ai,oi) = map(x -> last(x), get_weights(Bbao_data,weights_data,bi,ai,oi))


function POMDPs.solve(solver::X, model::POMDP) where X<:BIBSolver
    constants = get_constants(model)

    # 1: Precompute observation probabilities
    SAO_probs = []                                  # ∀s,a,o, gives probability of o given (s,a)
    SAOs = []                                       # ∀s,a, gives os with p>0
    SAO_probs, SAOs = get_all_obs_probs(model; constants)

    # 2 : Pre-compute all beliefs after 1 step
    B = []                                          # Our set of beliefs
    B_idx = []                                      # ∀s,a,o, gives index of b_sao in B
    B, B_idx = get_belief_set(model, SAOs; constants)

    # 3 : Compute Q-values bsoa beliefs
    Qs = get_QMDP_Beliefset(model, B, constants)    # ∀b∈B, contains QBIB value (initialized using QMDP)

    # 3.5 : combine all data thus far
    S_dict = Dict( zip(states(model), 1:length(states(model))))
    Data = BIB_Data(Qs, B,B_idx,SAO_probs,SAOs, S_dict, constants)

    # 4 : If WBIB or EBIB, precompute all beliefs after 2 steps
    Bbao_data = []
    if solver isa WBIBSolver || solver isa EBIBSolver
        Bbao_data = get_Bbao(model, Data, constants)
    end

    # 5 : If using EBIB, pre-compute entropy weights
    Weights = []
    if solver isa EBIBSolver
        Weights = get_entropy_weights_all(model,Data.B, Bbao_data)
    end

    # Lets be overly fancy! Define a function for computing Q, depending on the specific solver
    get_Q, args = identity, []
    if solver isa SBIBSolver
        pol, get_Q, args = SBIBPolicy, get_QBIB_Beliefset, (Data,)
    elseif solver isa WBIBSolver
        pol, get_Q, args = WBIBPolicy, get_QWBIB_Beliefset, (Data, Bbao_data)
    elseif solver isa EBIBSolver
        pol, get_Q, args = EBIBPolicy, get_QEBIB_Beliefset, (Data, Bbao_data, Weights)
    else
        throw("Solver type not recognized!")
    end
    
    # Now iterate:
    for i=1:solver.max_iterations
        # printdb(i)
        Qs, max_dif = get_Q(model, Qs, args...)
        max_dif < solver.precision && (printdb("breaking after $i iterations:"); break)
    end
    printdb(Qs)
    return pol(model, BIB_Data(Qs,Data))
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
        for ai in 1:na
            SAOs[si,ai] = []
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


function get_possible_obs(b::DiscreteHashedBelief, ai, SAOs, S_dict)
    possible_os = Set{Int}()
    for s in support(b)
        si = S_dict[s]
        union!(possible_os, SAOs[si,ai])
    end
    return collect(possible_os)
end

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
function get_Bbao(model, Data, constants)
    B = Data.B
    U = DiscreteHashedBeliefUpdater(model)
    nb = length(B)

    Bbao = []
    Bbao_idx = Array{Dict{Int, Tuple{Bool,Int}}}(undef, nb, constants.na)

    # Record bao: reference B if it's already in there, otherwise add to Bbao
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Bbao_idx[bi,ai] = Dict{Int, Tuple{Bool, Int}}()
            for (oi,o) in enumerate(constants.O)
                bao = POMDPs.update(U,b,a,o)
                if length(support(bao)) > 0
                    in_B = true
                    k = findfirst( x-> x==bao, B)
                    if isnothing(k)
                        in_B = false
                        k = findfirst(x -> x==bao, Bbao)
                        isnothing(k) && (push!(Bbao, bao); k=length(Bbao))
                    end
                    Bbao_idx[bi,ai][oi] = (in_B, k)
                end              
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
            have_overlap(b,bp) && push!(B_overlap[bi], bpi)
        end
    end
    # Record overlap for bao
    for (bi,b) in enumerate(Bbao)
        Bbao_overlap[bi] = []
        for (bpi,bp) in enumerate(B)
            have_overlap(b,bp) && push!(Bbao_overlap[bi], bpi)
        end
    end

    B_entropy = map( b -> get_entropy(b), B)

    return BBAO_Data(Bbao, Bbao_idx, B_overlap, Bbao_overlap, B_entropy)
end

function have_overlap(b,bp)
    # condition: bp does not contain any states not in support of b
    for sp in support(bp)
        pdf(b,sp) == 0 && (return false)
    end
    return true
end

function get_overlapping_beliefs(b, B::Vector)
    Bs, B_idxs = [], []
    for (bpi, bp) in enumerate(B)
        if have_overlap(b,bp)
            push!(Bs, bp)
            push!(B_idxs, bpi)
        end
    end
    return Bs, B_idxs
end

function get_entropy(b)
    entropy = 0
    for (s,p) in weighted_iterator(b)
        entropy += -log(p) * p
    end
    return entropy
    # return sum( (_s, prob) -> -prob * log(prob), weighted_iterator(b))
end

# Entropy stuff

function get_entropy_weights_all(model, B, Bbao_data::BBAO_Data)
    #TODO: only use those Bs that we need!
    B_weights = Array{Vector{Tuple{Int,Float64}}}(undef, length(B))
    Bbao_weights = Array{Vector{Tuple{Int,Float64}}}(undef, length(Bbao_data.Bbao))
    for (bi, b) in enumerate(B)
        # B_weights[bi] = get_entropy_weights(model,b, B; overlap=Bbao_data.B_overlap[bi])
        B_weights[bi] = get_entropy_weights(model,b, B; bi=(true,bi), Bbao_data=Bbao_data)
    end
    for (bi, b) in enumerate(Bbao_data.Bbao)
        # Bbao_weights[bi] = get_entropy_weights(model,b, B; overlap=Bbao_data.Bbao_overlap[bi])
        Bbao_weights[bi] = get_entropy_weights(model,b, B; bi=(false,bi), Bbao_data=Bbao_data)
    end
    return Weights_Data(B_weights, Bbao_weights)
end

function get_entropy_weights(model_pomdp::POMDP, b, B; bi=nothing, Bbao_data=nothing )
    B_relevant = []
    B_entropies = []
    B_idxs = []
    if !(Bbao_data isa Nothing) && !(bi isa Nothing)
        first(bi) ? (overlap=Bbao_data.B_overlap[last(bi)]) : (overlap=Bbao_data.Bbao_overlap[last(bi)])
        length(overlap) == 0 && ( printdb(b, bi))
        for bpi in overlap
            push!(B_relevant, B[bpi])
            push!(B_idxs, bpi)
            push!(B_entropies, Bbao_data.B_entropies[bpi])
        end
    else
        #TODO: still compute B_relevant.
        B_relevant = B
        B_idxs = 1:length(B)
        B_entropies = map( b -> get_entropy(b), B)
    end

    # model = direct_model(Gurobi.Optimizer(GRB_ENV))
    # set_silent(model)
    # set_attribute(model, "TimeLimit", 0.5)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0.0 <= b_ps[1:length(B_relevant)] <= 1.0)
    for s in states(model_pomdp)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B_relevant)
            p = pdf(bp,s)
            if p > 0
                push!(Idx, bpi)
                push!(Ps,p)
            end
        end
        length(Idx) > 0 && @constraint(model, sum(b_ps[Idx[i]] * Ps[i] for i in 1:length(Idx)) == pdf(b,s) )
    end
    @objective(model, Max, sum(b_ps.*B_entropies))
    optimize!(model)

    weights=[]
    cumprob = 0.0
    for (bpi,bp) in enumerate(B_relevant)
        prob = JuMP.value(b_ps[bpi])
        cumprob += prob
        real_bpi = B_idxs[bpi]
        push!(weights, (real_bpi, prob))
    end
    weights = map(x -> (first(x), last(x)/cumprob), weights)
    return(weights)
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

function get_QBIB_Beliefset(model::POMDP, Q, Data::BIB_Data)
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(Data.B)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QBIB_ba(model,b,a, Q, Data; ai=ai)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

function get_QBIB_ba(model::POMDP,b,a,Qs,B_idx,SAO_probs, SAOs, constants::C; ai=nothing, S_dict=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), actions(model)) )
    Q = breward(model,b,a)
    for (oi, o) in enumerate(constants.O)
        Qo = zeros(constants.na)
        for s in support(b)
            si = S_dict[s]
            if oi in SAOs[si,ai]
                p = pdf(b,s) * SAO_probs[oi,si,ai]
                bp_idx = B_idx[si,ai,oi]
                Qo = Qo .+ (p .* Qs[bp_idx,:])
            end
        end
        Q += discount(model) * maximum(Qo)
    end
    return Q
end
get_QBIB_ba(model::POMDP,b,a,Q,D::BIB_Data; ai=nothing) = get_QBIB_ba(model,b,a, Q, D.B_idx, D.SAO_probs, D.SAOs, D.constants; ai=ai, S_dict=D.S_dict)
get_QBIB_ba(model::POMDP,b,a,D::BIB_Data; ai=nothing) = get_QBIB_ba(model,b,a, D.Q, D.B_idx, D.SAO_probs, D.SAOs, D.constants; ai=ai, S_dict=D.S_dict)

############ WBIB ###################

function get_QWBIB_Beliefset(model::POMDP, Q, Data::BIB_Data, Bbao_data::BBAO_Data)
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (bi,b) in enumerate(Data.B)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[bi,ai] = get_QWBIB_ba(model,b,a, Q, Data; ai=ai, Bbao_data=Bbao_data, bi=bi)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

get_QWBIB_ba(model::POMDP, b,a,Q, D::BIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QWBIB_ba(model, b, a, Q, D.B, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)
get_QWBIB_ba(model::POMDP, b,a,D::BIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QWBIB_ba(model, b, a, D.Q, D.B, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)
function get_QWBIB_ba(model::POMDP,b,a,Qs,B, SAO_probs, constants::C; ai=nothing, Bbao_data=nothing, bi=nothing, S_dict=nothing)
    Q = breward(model,b,a)
    for (oi, o) in enumerate(constants.O)
        Qo = -Inf
        for (api, ap) in enumerate(constants.A) #TODO: would it be quicker to let the solver also find the best action?
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
    # model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
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
    @objective(model, Max, sum(Qs .* b_ps))
    optimize!(model)
    return(objective_value(model))
end

############ EBIB ###################

function get_QEBIB_Beliefset(model::POMDP,Q, Data::BIB_Data, Bbao_data, Weights)
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(Data.B)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QEBIB_ba(model,b,a, Q, Data; ai=ai, bi=b_idx, Bbao_data=Bbao_data, Weights_data=Weights)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

function get_QEBIB_ba(model::POMDP, b, a, Qs, B, SAOs, SAO_probs, constants::C; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing, S_dict=nothing)
    Q = breward(model,b,a)
    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        o = constants.O[oi]
        Qo = []
        if !(Bbao_data isa Nothing) && !(Weights_data isa Nothing) && !(bi isa Nothing)
            bao = get_bao(Bbao_data, bi, ai, oi, B)
            weights = get_weights_indexfree(Bbao_data, Weights_data,bi,ai,oi)
            this_Qs = Qs[get_overlap(Bbao_data, bi, ai, oi), :]
            Qo = sum(weights .* this_Qs, dims=1)
        else
            bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
            relevant_Bs, relevant_Bis = get_overlapping_beliefs(bao, B)
            weights = map(x -> last(x), get_entropy_weights(model,bao,relevant_Bs))
            this_Qs = Qs[relevant_Bis,:]
            Qo = sum(weights .* this_Qs, dims=1)
        end
        p = 0
        for s in support(b)
            p += pdf(b,s) * SAO_probs[oi,S_dict[s],ai]
        end
        # printdb(p, discount(model), maximum(Qo))
        Q += p * discount(model) * maximum(Qo)
    end
    return Q
end
get_QEBIB_ba(model::POMDP, b,a, D::BIB_Data; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing) = get_QEBIB_ba(model,b,a,D.Q,D.B, D.SAOs,D.SAO_probs, D.constants; ai=ai,bi=bi,Bbao_data=Bbao_data,Weights_data=Weights_data, S_dict=D.S_dict)
get_QEBIB_ba(model::POMDP, b,a, Q, D::BIB_Data; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing) = get_QEBIB_ba(model,b,a,Q,D.B, D.SAOs,D.SAO_probs, D.constants; ai=ai,bi=bi,Bbao_data=Bbao_data,Weights_data=Weights_data, S_dict=D.S_dict)

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

struct EBIBPolicy <: BIBPolicy
    model::POMDP
    Data::BIB_Data
end

POMDPs.action(π::X, b) where X<: BIBPolicy = first(action_value(π, b))
POMDPs.value(π::X, b) where X<: BIBPolicy = last(action_value(π,b))

@memoize LRU(maxsize=100) function action_value(π::SBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

@memoize LRU(maxsize=100) function action_value(π::WBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QWBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

@memoize LRU(maxsize=100) function action_value(π::EBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = get_QEBIB_ba(model, b, a, π.Data; ai=ai)
        # Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        # Qa = get_QWBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end




