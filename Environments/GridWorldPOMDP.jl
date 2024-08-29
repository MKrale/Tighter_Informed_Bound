#Copied gridworld definition from POMDPModels, but now as a POMDP.

module AMGridworlds
using POMDPs
using POMDPTools
using Distributions
using StaticArrays
using Random


export AMGridworld, FrozenLakeSmall, FrozenLakeLarge


const GWPos = SVector{2,Int}

"""
    SimpleGridWorld(;kwargs...)

Create a simple grid world MDP. Options are specified with keyword arguments.

# States and Actions
The states are represented by 2-element static vectors of integers. Typically any Julia `AbstractVector` e.g. `[x,y]` can also be used for arguments. Actions are the symbols `:up`, `:left`, `:down`, and `:right`.

# Keyword Arguments
- `size::Tuple{Int, Int}`: Number of cells in the x and y direction [default: `(10,10)`]
- `rewards::Dict`: Dictionary mapping cells to the reward in that cell, e.g. `Dict([1,2]=>10.0)`. Default reward for unlisted cells is 0.0
- `terminate_from::Set`: Set of cells from which the problem will terminate. Note that these states are not themselves terminal, but from these states, the next transition will be to a terminal state. [default: `Set(keys(rewards))`]
- `tprob::Float64`: Probability of a successful transition in the direction specified by the action. The remaining probability is divided between the other neighbors. [default: `0.7`]
- `discount::Float64`: Discount factor [default: `0.95`]
"""
Base.@kwdef struct AMGridWorld <: POMDP{GWPos, Symbol, GWPos}
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>3.0)
    terminate_from::Set{GWPos}      = Set(keys(rewards))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
end

FrozenLakeSmall = AMGridWorld(
    (4,4),
    Dict(GWPos(4,4)=>1),
    Set([GWPos(2,2), GWPos(1,4), GWPos(4,2), GWPos(4,3), GWPos(4,4)]),
    0.7,
    0.99
    )

FrozenLakeLarge = AMGridWorld(
    (10,10),
    Dict(GWPos(10,10)=>1),
    Set([   GWPos(1,4), GWPos(2,4), GWPos(3,7), GWPos(5,4), GWPos(5,8),
            GWPos(6,2), GWPos(6,5), GWPos(7,1), GWPos(7,3), GWPos(8,9),
            GWPos(9,3), GWPos(9,6), GWPos(10,8), GWPos(10,10)]),
    0.7,
    0.99
    )

measurecost = -0.1

# States

function POMDPs.states(mdp::AMGridWorld)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(mdp::AMGridWorld, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(mdp.size)[s...]
    else
        return prod(mdp.size) + 1 # TODO: Change
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end
Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::GWPos)
    if all(1 .<= s[1] .<= d.size)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])

POMDPs.initialstate(mdp::AMGridWorld) = Deterministic(GWPos(1,1))
# POMDPs.initialstate(mdp::AMGridWorld) = GWUniform(mdp.size)
# POMDPs.initialstate(mdp::AMGridWorld) = GWUniform( (3,3) )

# Actions

POMDPs.actions(mdp::AMGridWorld) = (:up, :down, :left, :right, :measure)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0), :measure=>GWPos(0,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :measure=>5)

POMDPs.actionindex(mdp::AMGridWorld, a::Symbol) = aind[a]


# Transitions

POMDPs.isterminal(m::AMGridWorld, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(mdp::AMGridWorld, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    elseif a == :measure
        return Deterministic(s)
    end

    # destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    # destinations[1] = s
    # probs = @MVector(zeros(length(actions(mdp))+1))
    destinations, probs = [], []

    
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        inbounds(mdp, dest) || (dest = s) # hit an edge and come back
        if dest in destinations
            probs[findfirst(==(dest), destinations)] += prob
        else
            push!(destinations, dest)
            push!(probs, prob)
        end

    end

    # return SparseCat(convert(SVector, destinations), convert(SVector, probs))
    return SparseCat(destinations, probs)
end

function inbounds(m::AMGridWorld, s::AbstractVector{Int})
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2]
end

# observations

POMDPs.observations(m::AMGridWorld) = POMDPs.states(m)
function POMDPs.observation(m::AMGridWorld, a, sp)
    a == :measure && return Deterministic(sp)
    return POMDPs.initialstate(m)
end

POMDPs.obsindex(m::AMGridWorld, o) = POMDPs.stateindex(m,o)

# Rewards

POMDPs.reward(mdp::AMGridWorld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::AMGridWorld, s::AbstractVector{Int}, a::Symbol) = a==:measure ? (return measurecost) : return(reward(mdp, s))


# discount

POMDPs.discount(mdp::AMGridWorld) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::AMGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::AMGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end

end