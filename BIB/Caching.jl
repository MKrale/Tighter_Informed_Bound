struct C
    S; A; O 
    ns; na; no
end
get_constants(model) = C( states(model), actions(model), observations(model),
                         length(states(model)), length(actions(model)), length(observations(model)))


#########################################
#             BIB_Data:
#########################################

struct BIB_Data
    Q::Union{Array{Float64,2}, Nothing}
    B::Vector
    B_idx::Array{Int,3}
    SAO_probs::Array{Float64,3}
    SAOs::Array{Vector{Int},2}
    S_dict::Dict{Any, Int}
    constants::C
end
BIB_Data(Q::Array{Float64,2}, D::BIB_Data) = BIB_Data(Q,D.B, D.B_idx, D.SAO_probs, D.SAOs, D.S_dict, D.constants)

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

    for (oi,o) in enumerate(constants.O)
        for (si,s) in enumerate(constants.S)
            for (ai,a) in enumerate(constants.A)
                SAO_probs[oi,si,ai] = get_obs_prob(model,o,s,a)
                SAO_probs[oi,si,ai] > 0.0 && push!(SAOs[si,ai], oi)
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
function get_obs_prob(model::POMDP, o, b::DiscreteHashedBelief, a) 
    sum = 0
    for (s,p) in zip(b.state_list, b.probs)
        sum += p*get_obs_prob(model,o,s,a)
    end
    return sum
end

function get_possible_obs(bi::Tuple{Bool, Int}, ai, Data, Bbao_data)
    if first(bi)
        b = Data.B[last(bi)]
        possible_os = Set{Int}
        for s in support(b)
            si = S_dict[s]
            union!(possible_os, SAOs[si,ai])
        end
        return 
    else
        return collect(keys(Bbao_Data.Bbao_idx[last(bi), ai]))
    end
end

function get_possible_obs(b::DiscreteHashedBelief, ai, SAOs, S_dict)
    possible_os = Set{Int}()
    for s in support(b)
        si = S_dict[s]
        union!(possible_os, SAOs[si,ai])
    end
    return collect(possible_os)
end

function get_belief_set(model, SAOs; constants::Union{C,Nothing}=nothing)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no
    U = DiscreteHashedBeliefUpdater(model)

    B = Array{DiscreteHashedBelief,1}()       
    B_idx = zeros(Int,ns,na,no)

    # Initialize with states and initial belief
    for s in S 
        push!(B, DiscreteHashedBelief([s],[1.0]))
    end
    b_init = DiscreteHashedBelief(initialstate(model))
    k = findfirst( x -> x==b_init , B)
    if isnothing(k)
        push!(B,b_init)
    end

    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief([s],[1.0])
        for (ai,a) in enumerate(A)
            for oi in SAOs[si,ai]
                o = O[oi]
                b = update(U, b_s, a, o)
                k = findfirst( x -> x==b , B)
                if isnothing(k)
                    push!(B,b)
                    k=length(B)
                end
                B_idx[si,ai,oi] = k
            end
        end
    end

    return B, B_idx
end

#########################################
#          Bbao_data:
#########################################

struct BBAO_Data
    Bbao::Vector
    Bbao_idx::Array{Dict{Int,Tuple{Bool,Int}},2}
    B_in_Bbao::BitVector
    B_overlap::Array{Vector{Int}}
    Bbao_overlap::Array{Vector{Int}}
    B_entropies::Array{Float64}
    Valid_weights::Array{Dict{Int,Float64}}
end

function get_bao(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int, B)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return B[baoi]) : (return Bbao_data.Bbao[baoi])
end
function get_overlap(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return Bbao_data.B_overlap[baoi]) : (return Bbao_data.Bbao_overlap[baoi])
end

function get_Bbao(model, Data, constants)
    B = Data.B
    S_dict = Data.S_dict
    O_dict = Dict( zip(constants.O, 1:constants.no) )
    U = DiscreteHashedBeliefUpdater(model)
    nb = length(B)

    Bbao = []
    B_in_Bboa = zeros(Bool, length(B))
    Bbao_valid_weights = Dict{Int,Float64}[]
    Bbao_idx = Array{Dict{Int, Tuple{Bool,Int}}}(undef, nb, constants.na)

    Bs_found = Dict(zip(B, map( idx -> (true, idx), 1:length(B))))

    Bs_lowest_support_state = Dict()
    for (bi, b) in enumerate(B)
        s_lowest = minimum(s -> stateindex(model, s), support(b))
        if haskey(Bs_lowest_support_state, s_lowest)
            push!(Bs_lowest_support_state[s_lowest], bi)
        else
            Bs_lowest_support_state[s_lowest] = [bi]
        end
    end

    # Record bao: reference B if it's already in there, otherwise add to Bbao
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Bbao_idx[bi,ai] = Dict{Int, Tuple{Bool, Int}}()
            for oi in get_possible_obs(b,ai,Data.SAOs,S_dict)
                o = constants.O[oi]
                bao = POMDPs.update(U,b,a,o)
                if length(support(bao)) > 0
                    if haskey(Bs_found, bao)
                        (in_B, idx) = Bs_found[bao]
                        in_B && (B_in_Bboa[idx] = true)
                        Bbao_idx[bi,ai][oi] = (in_B, idx)
                    else
                        push!(Bbao, bao)
                        k=length(Bbao)
                        valid_weights = Dict{Int, Float64}()
                        for (s,ps) in weighted_iterator(b)
                            valid_weights[Data.B_idx[S_dict[s],ai,oi]] = ps
                        end
                        push!(Bbao_valid_weights, valid_weights)
                        Bbao_idx[bi,ai][oi] = (false, k)
                        Bs_found[bao] = (false, k)
                    end
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
        s_lowest = minimum(map(s -> stateindex(model, s), support(b)))
        s_highest = maximum(map(s -> stateindex(model, s), support(b)))
        for s=s_lowest:s_highest
            if haskey(Bs_lowest_support_state, s)
                for bpi in Bs_lowest_support_state[s]
                    have_overlap(b,B[bpi]) && push!(B_overlap[bi], bpi)
                end
            end
        end
    end
    # Record overlap for bao
    for (bi,b) in enumerate(Bbao)
        Bbao_overlap[bi] = []
        s_lowest = minimum(map(s -> stateindex(model, s), support(b)))
        s_highest = maximum(map(s -> stateindex(model, s), support(b)))
        for s=s_lowest:s_highest
            if haskey(Bs_lowest_support_state, s)
                for bpi in Bs_lowest_support_state[s]
                    have_overlap(b,B[bpi]) && push!(Bbao_overlap[bi], bpi)
                end
            end
        end
    end

    B_entropy = map( b -> get_entropy(b), B)
    return BBAO_Data(Bbao, Bbao_idx, B_in_Bboa, B_overlap, Bbao_overlap, B_entropy, Bbao_valid_weights)
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

#########################################
#               Weights:
#########################################

struct Weights_Data
    B_weights::Vector{Vector{Tuple{Int,Float64}}}
    Bbao_weights::Vector{Vector{Tuple{Int,Float64}}}
end
function get_weights(Bbao_data, weights_data, bi, ai, oi)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return weights_data.B_weights[baoi]) : (return weights_data.Bbao_weights[baoi])
end
get_weights_indexfree(Bbao_data, weights_data,bi,ai,oi) = map(x -> last(x), get_weights(Bbao_data,weights_data,bi,ai,oi))

function get_entropy_weights_all(B, Bbao_data::BBAO_Data)
    #TODO: only use those Bs that we need!
    B_weights = Array{Vector{Tuple{Int,Float64}}}(undef, length(B))
    Bbao_weights = Array{Vector{Tuple{Int,Float64}}}(undef, length(Bbao_data.Bbao))
    for (bi, b) in enumerate(B)
        if Bbao_data.B_in_Bbao[bi]
            # B_weights[bi] = get_entropy_weights(model,b, B; overlap=Bbao_data.B_overlap[bi])
            B_weights[bi] = get_entropy_weights(b, B; bi=(true,bi), Bbao_data=Bbao_data)
        end
    end
    for (bi, b) in enumerate(Bbao_data.Bbao)
        # Bbao_weights[bi] = get_entropy_weights(model,b, B; overlap=Bbao_data.Bbao_overlap[bi])
        Bbao_weights[bi] = get_entropy_weights(b, B; bi=(false,bi), Bbao_data=Bbao_data)
    end
    return Weights_Data(B_weights, Bbao_weights)
end
   
function get_entropy_weights(b, B; bi=nothing, Bbao_data=nothing )
    B_relevant = []
    B_start = []
    B_entropies = []
    B_idxs = []
    if !(Bbao_data isa Nothing) && !(bi isa Nothing)
        if first(bi)
            overlap=Bbao_data.B_overlap[last(bi)]
            valid_weights = Dict(last(bi) => 1.0)
        else
            overlap=Bbao_data.Bbao_overlap[last(bi)]
            valid_weights = Bbao_data.Valid_weights[last(bi)]
        end
        for bpi in overlap
            push!(B_relevant, B[bpi])
            push!(B_idxs, bpi)
            push!(B_entropies, Bbao_data.B_entropies[bpi])
            haskey(valid_weights, bpi) ? push!(B_start, valid_weights[bpi]) : push!(B_start, 0.0)
        end
    else
        B_relevant = B
        B_idxs = 1:length(B)
        B_entropies = map( b -> get_entropy(b), B)
    end
    model = direct_generic_model(Float64,Gurobi.Optimizer(GRB_ENV))
    # model = direct_generic_model(Float64,Tulip.Optimizer())
    # model = Model(Tulip.Optimizer; add_bridges = false)
    # model = direct_model(HiGHS.Optimizer())
    # model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_string_names_on_creation(model, false)
    @variable(model, 0.0 <= b_ps[1:length(B_relevant)] <= 1.0)
    # !(B_start == []) && set_start_value.(b_ps, B_start)
    for s in support(b)
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