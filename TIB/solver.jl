#########################################
#               Solver:
#########################################

abstract type TIBSolver <: Solver end

@kwdef struct STIBSolver <: TIBSolver
    max_iterations::Int64   = 250       # maximum iterations taken by solver
    max_time::Float64       = 3600      # maximum time spent solving
    precision::Float64      = 1e-4      # precision at which iterations is stopped
    precomp_solver          = FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600)
end

@kwdef struct ETIBSolver <: TIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
    # precomp_solver          = FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600)
 end

 @kwdef struct OTIBSolver <: TIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
    # precomp_solver          = FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600)
 end

verbose = false
POMDPs.solve(solver::X, model::POMDP) where X<: TIBSolver = solve(solver, model; Data=nothing)
function solve(solver::X, model::POMDP; Data::Union{TIB_Data,Nothing}=nothing) where X<:TIBSolver
    t0 = time()

    if isnothing(Data) # This is the default case: 
        constants = get_constants(model)

        # 1: Precompute observation probabilities
        # SAO_probs = []                                  # ∀ s,a,o, gives probability of o given (s,a)
        # SAOs = []                                       # ∀ s,a, gives os with p>0
        SAO_probs, SAOs = get_all_obs_probs(model; constants)
        S_dict = Dict( zip(constants.S, 1:constants.ns))

        # 2 : Pre-compute all beliefs after 1 step
        # B = []                                          # Our set of beliefs
        # B_idx = []                                      # ∀ s,a,o, gives index of b_sao in B
        # Br = []                                         # ∀ b∈B,a, gives expected reward                                      
        B, B_idx = get_belief_set(model, SAOs; constants)
        Br = get_Br(model, B, constants)

        Data = TIB_Data(nothing, B,B_idx, Br, SAO_probs,SAOs, S_dict, constants)
    end

    # 3 : Compute Q-values bsoa beliefs
    Qs = precompute_Qs(model, Data, solver.precomp_solver)    # ∀ b ∈ B, contains QTIB value (initialized using QMDP)

    # 4 : If OTIB or ETIB, precompute all beliefs after 2 steps
    Bbao_data = []
    if solver isa OTIBSolver || solver isa ETIBSolver
        Bbao_data = get_Bbao(model, Data, Data.constants)
    end
    t_init = time() - t0
    verbose && printdb("general init time:", t_init)

    # 5 : If using ETIB, pre-compute entropy weights
    
    Weights = []
    if solver isa ETIBSolver
        Weights = get_entropy_weights_all(Data.B, Bbao_data)
    end

    t_w = time() - t0 - t_init
    verbose && printdb("weights calculation time:", t_w)

    # Lets be overly fancy! Define a function for computing Q, depending on the specific solver
    get_Q, args = identity, []
    if solver isa STIBSolver
        pol, get_Q, args = STIBPolicy, get_QTIB_Beliefset, (Data,)
    elseif solver isa OTIBSolver
        pol, get_Q, args = OTIBPolicy, get_QOTIB_Beliefset, (Data, Bbao_data)
    elseif solver isa ETIBSolver
        pol, get_Q, args = ETIBPolicy, get_QETIB_Beliefset, (Data, Bbao_data, Weights)
    else
        throw("Solver type not recognized!")
    end
    
    # Now iterate:
    it = 0
    factor = discount(model) / (1-discount(model))
    for i=1:solver.max_iterations
        time_left = solver.max_time-(time()-t0)
        verbose && printdb("starting iteration $it")
        Qs, max_dif = get_Q(model, Qs,time_left, args...)
        if factor * max_dif < solver.precision || time()-t0 > solver.max_time
            break
        end
        it = i
    end
    t_it = time()- t0 - t_init - t_w
    verbose && printdb("iteration time $t_it (avg over $it iterations: $(t_it/it))")
    return pol(model, TIB_Data(Qs,Data))
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
                    Qs[b_idx, ai] += pdf(b,s) * π_QMDP.Data.Q[si,ai]
                end
            end
        end
    end
    return Qs
end

function precompute_Qs(model::POMDP, Data::TIB_Data, solver::X ; getdata=false) where X<: Solver

	π = solve(solver, model; Data=Data)
    B, constants = Data.B, Data.constants
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (si, s) in enumerate(constants.S)
                if pdf(b,s) > 0
                    Qs[b_idx, ai] += pdf(b,s) * π.Data.Q[si,ai]
                end
            end
        end
    end
    getdata ? (return TIB_Data(Qs, Data)) : (return Qs)
end

############ TIB ###################

function get_QTIB_Beliefset(model::POMDP, Q, timeleft, Data::TIB_Data)
    t0 = time()
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QTIB_ba(model,b,a, Q, Data; bi=b_idx, ai=ai)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

function get_QTIB_ba(model::POMDP,b,a,Qs,B_idx,Br,SAO_probs, SAOs, constants::C; ai=nothing, bi=nothing, S_dict=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), constants.A) )
    isnothing(bi) ? (Q = breward(model,b,a)) : (Q = Br[bi,ai])
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
get_QTIB_ba(model::POMDP,b,a,Q,D::TIB_Data; bi=nothing, ai=nothing) = get_QTIB_ba(model,b,a, Q, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.constants; bi=bi, ai=ai, S_dict=D.S_dict)
get_QTIB_ba(model::POMDP,b,a,D::TIB_Data; bi=nothing, ai=nothing) = get_QTIB_ba(model,b,a, D.Q, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.constants; bi=bi, ai=ai, S_dict=D.S_dict)

############ OTIB ###################

function get_QOTIB_Beliefset(model::POMDP, Q, timeleft, Data::TIB_Data, Bbao_data::BBAO_Data)
    t0 = time()
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (bi,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[bi,ai] = get_QOTIB_ba(model,b,a, Q, Data; ai=ai, Bbao_data=Bbao_data, bi=bi)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

get_QOTIB_ba(model::POMDP, b,a,Q, D::TIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QOTIB_ba(model, b, a, Q, D.B, D.Br, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)
get_QOTIB_ba(model::POMDP, b,a,D::TIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QOTIB_ba(model, b, a, D.Q, D.B, D.Br, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)

function get_QOTIB_ba(model::POMDP,b,a,Qs,B,Br, SAOs, SAO_probs, constants::C; ai=nothing, Bbao_data=nothing, bi=nothing, S_dict=nothing)
    
    opt_model = Model(Clp.Optimizer; add_bridges=false)
    # opt_model = direct_generic_model(Float64, HiGHS.Optimizer())
    set_silent(opt_model)
    set_string_names_on_creation(opt_model, false)
    
    isnothing(bi) ? (Q = breward(model,b,a)) : (Q = Br[bi,ai])
    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        Qo = -Inf
        for (api, ap) in enumerate(constants.A) #TODO: would it be quicker to let the solver also find the best action?
            thisQo = 0
            if !(Bbao_data isa Nothing) && !(bi isa Nothing)
                bao = get_bao(Bbao_data, bi, ai, oi, B)
                overlap_idxs = get_overlap(Bbao_data, bi, ai, oi)
                thisBs, thisQs = B[overlap_idxs], Qs[overlap_idxs, api]
                empty!(opt_model)
                thisQo = get_QLP(bao, thisQs, thisBs, opt_model)
            else
                o = constants.O[oi]
                bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
                B_rel, Bidx_rel = get_overlapping_beliefs(bao,B)
                empty!(opt_model)
                thisQo = get_QLP(bao, Qs[Bidx_rel,api],B_rel, opt_model)
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
        # model = Model(HiGHS.Optimizer)
        model = Model(Clp.Optimizer; add_bridges=false)
        set_silent(model)
        set_string_names_on_creation(opt_model, false)
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

############ ETIB ###################

function get_QETIB_Beliefset(model::POMDP,Q, timeleft, Data::TIB_Data, Bbao_data, Weights)
    Qs_new = zero(Q) # TODO: this may be inefficient?
    t0 = time()
    for (b_idx,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QETIB_ba(model,b,a, Q, Data; ai=ai, bi=b_idx, Bbao_data=Bbao_data, Weights_data=Weights)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

function get_QETIB_ba(model::POMDP, b, a, Qs, B, Br, SAOs, SAO_probs, constants::C; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing, S_dict=nothing)
    isnothing(bi) ? (Q = breward(model,b,a)) : (Q = Br[bi,ai])
    (bi isa Nothing || Bbao_data isa Nothing) ? (Os = collect(keys(get_possible_obs_probs(b,ai,SAOs,SAO_probs,S_dict)))) : (Os = get_possible_obs( (true,bi) ,ai,SAOs,Bbao_data))
    for oi in Os
        o = constants.O[oi]
        Qo = []
        p = 0
        if !(Bbao_data isa Nothing) && !(Weights_data isa Nothing) && !(bi isa Nothing)
            bao = get_bao(Bbao_data, bi, ai, oi, B)
            w = discount(model) * Bbao_data.BAO_probs[oi,bi,ai]
            Q += w * maximum( sum(get_weights_indexfree(Bbao_data, Weights_data,bi,ai,oi) .* Qs[get_overlap(Bbao_data, bi, ai, oi), :], dims=1))
            ### Or written out:
            # weights = get_weights_indexfree(Bbao_data, Weights_data,bi,ai,oi)
            # this_Qs = Qs[get_overlap(Bbao_data, bi, ai, oi), :]
            # Qo = sum(weights .* this_Qs, dims=1)
            # p = Bbao_data.BAO_probs[oi,bi,ai]
        else
            bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
            relevant_Bs, relevant_Bis = get_overlapping_beliefs(bao, B)
            weights = map(x -> last(x), get_entropy_weights(bao,relevant_Bs))
            this_Qs = Qs[relevant_Bis,:]
            Qo = sum(weights .* this_Qs, dims=1)
            for s in support(b)
                p += pdf(b,s) * SAO_probs[oi,S_dict[s],ai]
            end
            Q += p * discount(model) * maximum(Qo)
        end
    end
    return Q
end
get_QETIB_ba(model::POMDP, b,a, D::TIB_Data; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing) = get_QETIB_ba(model,b,a,D.Q,D.B, D.Br, D.SAOs,D.SAO_probs, D.constants; ai=ai,bi=bi,Bbao_data=Bbao_data,Weights_data=Weights_data, S_dict=D.S_dict)
get_QETIB_ba(model::POMDP, b,a, Q, D::TIB_Data; ai=nothing, bi=nothing, Bbao_data=nothing, Weights_data=nothing) = get_QETIB_ba(model,b,a,Q,D.B, D.Br, D.SAOs,D.SAO_probs, D.constants; ai=ai,bi=bi,Bbao_data=Bbao_data,Weights_data=Weights_data, S_dict=D.S_dict)

#########################################
#               Policy:
#########################################

abstract type TIBPolicy <: Policy end
POMDPs.updater(π::X) where X<:TIBPolicy = DiscreteHashedBeliefUpdater(π.model)

struct STIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end

struct OTIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end

struct ETIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end

POMDPs.action(π::X, b) where X<: TIBPolicy = first(action_value(π, b))
POMDPs.value(π::X, b) where X<: TIBPolicy = last(action_value(π,b))

function action_value(π::STIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QTIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::OTIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QOTIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::ETIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        # if π.evaluate_full
            Qa = get_QETIB_ba(model, b, a, π.Data; ai=ai)
        # else
            # Qa = get_QTIB_ba(model, b, a, π.Data; ai=ai)
        # end
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function get_heuristic_pointset(policy::X; get_only_Bs=false) where X<:TIBPolicy
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
    # get_only_Bs && return corner_values
    return corner_values, B_heuristic[ns:end], V_heuristic[ns:end]
end
