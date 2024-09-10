#########################################
#          Belief Definitions
#########################################

struct DiscreteHashedBelief{S}
    state_list::Vector{S}       # assumed sorted!
    probs::Vector{Float64}
    hash::UInt
end

function DiscreteHashedBelief(state_list::Vector, probs::Vector{<:Float64})
    nonzero_els = findall(>(0),probs)
    state_list, probs = state_list[nonzero_els], probs[nonzero_els]
    idxs = sortperm(state_list; lt= (x,y) -> objectid(x) < objectid(y))
    ordered_state_list, ordered_probs = state_list[idxs], probs[idxs]
    hash = makeDBhash(ordered_state_list, ordered_probs)
    return DiscreteHashedBelief(ordered_state_list, ordered_probs, hash)
end


#TODO: this is bad: we never check if our variable is indeed a belief. 
# I don't see any way to do this though: the beliefs used throughout the POMDP framework do not have a consistent supertype (even though they should all be distributions...)
# Maybe checking for the existance of a support/pdf function would be enough, but the way of doing this in Julia (method_exists()) seems to be removed and is the only thing I can find.
function DiscreteHashedBelief(b) 
    S,P = [], Float64[]
    for (s,p) in weighted_iterator(b)
        if p>0
            push!(S,s)
            push!(P,p)
        end
    end
    return DiscreteHashedBelief(S,P)
end


function POMDPs.rand(rng::AbstractRNG, s::Random.SamplerTrivial{DiscreteHashedBelief})
    d = s[]
    r = rand(rng)
    tot = 0.0
    for x in support(d)
        tot += pdf(d,x)
        r < tot && return x
    end
    tot < 1.0 && throw("Trying to sample from non-normalized belief (with total probability $tot)")
    throw("Error: sampling from DiscretizedBelief failed for unknown reason.")
end

function POMDPs.pdf(d::DiscreteHashedBelief, s) 
    k=findfirst( ==(s), d.state_list)               # This could use the fact that states are sorted...
    isnothing(k) ? (return 0) : (return d.probs[k])
end
# pdf_real(d::DiscretizedBelief,s) = haskey(d.b, s) ? last(d.b[s]) : 0
POMDPs.support(d::DiscreteHashedBelief) = d.state_list

# Base.iterate(d::DiscretizedBelief) = map( (x,s) -> iterate(d.b)
# Base.iterate(d::DiscretizedBelief, state) = iterate(d.b, state)
Base.length(d::DiscreteHashedBelief) = length(d.state_list)
# Base.eltype(d::DiscretizedBelief{Dict{V,Float64},Float64}) =  Pair{V, Float64, Float64}

mean(d::DiscreteHashedBelief) = throw("Function not implemented")
mode(d::DiscreteHashedBelief) = throw("Function not implemented")

function POMDPTools.beliefvec(model::POMDP,b::DiscreteHashedBelief)
    bv = zeros(length(states(model)))
    for (s, ps) in weighted_iterator(b)
        bv[s] = ps
    end
    return bv
end

function to_sparse_vector(model::POMDP, b::DiscreteHashedBelief)
    spv = spzeros(length(states(model)))
    for (si, s) in enumerate(b.state_list)
        spv[POMDPs.stateindex(model, s)] = b.probs[si]
    end
    return spv
end

#########################################
#          Hashing & stuff
#########################################

Base.:(==)(x::DiscreteHashedBelief, y::DiscreteHashedBelief) = (x.hash == y.hash) && all( map( s -> isapprox( pdf(x,s), pdf(y,s); atol=10^-3 ),  collect(support(x))))

function Base.:(<)(x::DiscreteHashedBelief, y::DiscreteHashedBelief)
    (x.hash != y.hash) && return (x.hash < y.hash)
    for k in sort(vcat(collect(support(x)), collect(support(y))))
        pdf(x,k) < pdf(y,k) && return true
        pdf(x,k) > pdf(y,k) && return false
    end
    return false
end
Base.isless(x::DiscreteHashedBelief{<:Any}, y::DiscreteHashedBelief{<:Any}) = x < y

makeDBhash(states_list::Vector, probs::Vector{Float64}) = hash(hash_alt(states_list), hash_alt(probs))
hash_alt(v::Vector) = foldr( (x,y) -> hash(x,y), v; init=UInt(0))

Base.hash(x::DiscreteHashedBelief, h::UInt) = hash(x.hash, h)
Base.hash(x::DiscreteHashedBelief) = hash(x,UInt(0))

#########################################
#          Belief Updater
#########################################

struct DiscreteHashedBeliefUpdater <: Updater
    model::POMDP
end

function initialize_belief(bu::DiscreteHashedBeliefUpdater, d)
    S,P = [], []
    for (s,p) in weighted_iterator(d)
        push!(S,s); push!(P,p)
    end
    return DiscreteHashedBelief(S,P)
end

function POMDPs.update(bu::DiscreteHashedBeliefUpdater, b::DiscreteHashedBelief,a,o)
    model = bu.model
    bnext = Dict{Any, Float64}()

    for (s, ps) in weighted_iterator(b)
        ss_next = transition(model, s, a)
        for (snext, psnext) in weighted_iterator(ss_next)
            po = obs_weight(model,s,a,snext,o)
            add_to_dict!(bnext, snext, ps*psnext*po)
        end
    end

    states, probs = collect(keys(bnext)), collect(values(bnext))
    probs ./= sum(probs)
    return DiscreteHashedBelief(states,probs)
end

#TODO: again, this is really bad, we never type-check b...
POMDPs.update(bu::DiscreteHashedBeliefUpdater, b, a, o) = update(bu, DiscreteHashedBelief(b),a,o) 