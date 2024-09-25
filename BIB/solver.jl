#########################################
#               Solver:
#########################################

abstract type BIBSolver <: Solver end

@kwdef struct SBIBSolver <: BIBSolver
    max_iterations::Int64   = 250       # maximum iterations taken by solver
    max_time::Float64       = 3600      # maximum time spent solving
    precision::Float64      = 1e-4      # precision at which iterations is stopped
end
@kwdef struct WBIBSolver <: BIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
 end

@kwdef struct EBIBSolver <: BIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
 end


verbose = false
function POMDPs.solve(solver::X, model::POMDP) where X<:BIBSolver
    t0 = time()
    constants = get_constants(model)

    # 1: Precompute observation probabilities
    SAO_probs = []                                  # ∀s,a,o, gives probability of o given (s,a)
    SAOs = []                                       # ∀s,a, gives os with p>0
    SAO_probs, SAOs = get_all_obs_probs(model; constants)
    S_dict = Dict( zip(constants.S, 1:constants.ns))

    # 2 : Pre-compute all beliefs after 1 step
    B = []                                          # Our set of beliefs
    B_idx = []                                      # ∀s,a,o, gives index of b_sao in B
    B, B_idx = get_belief_set(model, SAOs; constants)

    # 3 : Compute Q-values bsoa beliefs
    Data = BIB_Data(nothing, B,B_idx,SAO_probs,SAOs, S_dict, constants)
    Qs = get_FIB_Beliefset(model, Data, solver)    # ∀b∈B, contains QBIB value (initialized using QMDP)

    # 4 : If WBIB or EBIB, precompute all beliefs after 2 steps
    Bbao_data = []
    if solver isa WBIBSolver || solver isa EBIBSolver
        Bbao_data = get_Bbao(model, Data, constants)
    end
    t_init = time() - t0
    verbose && printdb("general init time:", t_init)

    # 5 : If using EBIB, pre-compute entropy weights
    
    Weights = []
    if solver isa EBIBSolver
        Weights = get_entropy_weights_all(Data.B, Bbao_data)
    end

    t_w = time() - t0 - t_init
    verbose && printdb("weights calculation time:", t_w)

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
    it = 0
    for i=1:solver.max_iterations
        Qs, max_dif = get_Q(model, Qs, args...)
        if max_dif < solver.precision || time()-t0 > solver.max_time
            break
        end
        it = i
    end
    t_it = time()- t0 - t_init - t_w
    verbose && printdb("iteration time $t_it (avg over $it iterations: $(t_it/it))")
    return pol(model, BIB_Data(Qs,Data))
end

#########################################
#            Value Computations:
#########################################

############ FIB & QMDP ###################

function get_QMDP_Beliefset(model::POMDP, B::Vector; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")

    π_QMDP = solve(QMDPSolver_alt(), model)
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (si, s) in enumerate(constants.S)
                if pdf(b,s) > 0
                    Qs[b_idx, ai] += pdf(b,s) * π_QMDP.Q[si,ai]
                end
            end
        end
    end
    return Qs
end

function get_FIB_Beliefset(model::POMDP, Data::BIB_Data, solver::X ; getdata=false) where X<: BIBSolver

	π = solve(FIBSolver_alt(precision=solver.precision, max_time=solver.max_time, max_iterations=solver.max_iterations*4), model) #; Data=Data)
    B, constants = Data.B, Data.constants
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (si, s) in enumerate(constants.S)
                if pdf(b,s) > 0
                    Qs[b_idx, ai] += pdf(b,s) * π.Q[si,ai]
                end
            end
        end
    end
    getdata ? (return BIB_Data(Qs, Data)) : (return Qs)
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
    isnothing(ai) && ( ai=findfirst(==(a), constants.A) )
    Q = breward(model,b,a)
    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        o = constants.O[oi]
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

get_QWBIB_ba(model::POMDP, b,a,Q, D::BIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QWBIB_ba(model, b, a, Q, D.B, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)
get_QWBIB_ba(model::POMDP, b,a,D::BIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QWBIB_ba(model, b, a, D.Q, D.B, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)

function get_QWBIB_ba(model::POMDP,b,a,Qs,B, SAOs, SAO_probs, constants::C; ai=nothing, Bbao_data=nothing, bi=nothing, S_dict=nothing)
    
    opt_model = Model(Clp.Optimizer; add_bridges=false)
    # opt_model = direct_generic_model(Float64, HiGHS.Optimizer())
    set_silent(opt_model)
    set_string_names_on_creation(opt_model, false)
    
    Q = breward(model,b,a)
    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        Qo = -Inf
        for (api, ap) in enumerate(constants.A) #TODO: would it be quicker to let the solver also find the best action?
            thisQo = 0
            if !(Bbao_data isa Nothing) && !(bi isa Nothing)
                bao = get_bao(Bbao_data, bi, ai, oi, B)
                overlap_idxs = get_overlap(Bbao_data, bi, ai, oi)
                thisBs, thisQs = B[overlap_idxs], Qs[overlap_idxs, api]
                empty!(opt_model)
                thisQo = get_QLP(bao, thisQs, thisBs, nothing)
            else
                o = constants.O[oi]
                bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
                B_rel, Bidx_rel = get_overlapping_beliefs(bao,B)
                empty!(opt_model)
                thisQo = get_QLP(bao, Qs[Bidx_rel,api],B_rel, nothing)
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
function get_QLP(b,Qs,B, model)
    if model isa Nothing
        model = Model(HiGHS.Optimizer)
        set_silent(model)
    end
    @variable(model, 0.0 <= b_ps[1:length(B)] <= 1.0)
    for s in support(b)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B)
            p = pdf(bp,s)
            if p > 0
                push!(Idx, bpi)
                push!(Ps,p)
            end
        end
        length(Idx) > 0 && @constraint(model, sum(b_ps[Idx[i]] * Ps[i] for i in 1:length(Idx)) == pdf(b,s) )
    end
    @objective(model, Min, sum(Qs .* b_ps))
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
    (bi isa Nothing || Bbao_data isa Nothing) ? (Os = collect(keys(get_possible_obs_probs(b,ai,SAOs,SAO_probs,S_dict)))) : (Os = get_possible_obs( (true,bi) ,ai,SAOs,Bbao_data))
    for oi in Os
        o = constants.O[oi]
        Qo = []
        p = 0
        if !(Bbao_data isa Nothing) && !(Weights_data isa Nothing) && !(bi isa Nothing)
            bao = get_bao(Bbao_data, bi, ai, oi, B)
            weights = get_weights_indexfree(Bbao_data, Weights_data,bi,ai,oi)
            this_Qs = Qs[get_overlap(Bbao_data, bi, ai, oi), :]
            Qo = sum(weights .* this_Qs, dims=1)
            p = Bbao_data.BAO_probs[oi,bi,ai]
        else
            bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
            relevant_Bs, relevant_Bis = get_overlapping_beliefs(bao, B)
            weights = map(x -> last(x), get_entropy_weights(bao,relevant_Bs))
            this_Qs = Qs[relevant_Bis,:]
            Qo = sum(weights .* this_Qs, dims=1)
            for s in support(b)
                p += pdf(b,s) * SAO_probs[oi,S_dict[s],ai]
            end
        end
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

function action_value(π::SBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::WBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QWBIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::EBIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        # if π.evaluate_full
            Qa = get_QEBIB_ba(model, b, a, π.Data; ai=ai)
        # else
            # Qa = get_QBIB_ba(model, b, a, π.Data; ai=ai)
        # end
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function get_heuristic_pointset(policy::X) where X<:BIBPolicy
    B_heuristic, V_heuristic = Vector{Float64}[], Float64[]
    ns = policy.Data.constants.ns
    S_dict = policy.Data.S_dict

    badBs = []

    for (b_idx, b) in enumerate(policy.Data.B)
        b_svector = spzeros(ns)
        for (s, p) in weighted_iterator(b)
            b_svector[stateindex(policy.model, s)] = p 
        end
        push!(B_heuristic, b_svector)
        push!(V_heuristic, maximum(policy.Data.Q[b_idx,:]))
    end
    corner_values = V_heuristic[1:ns]
    return corner_values, B_heuristic[ns:end], V_heuristic[ns:end]
end
