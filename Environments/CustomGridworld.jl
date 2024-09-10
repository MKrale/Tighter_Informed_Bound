#Copied gridworld definition from POMDPModels, but now as a POMDP.

module CustomGridWorlds
using POMDPs
using POMDPTools
using Distributions
using StaticArrays
using Random
using Memoization, LRUCache


export Gridworld, FrozenLakeSmall, FrozenLakeLarge, CustomMiniHallway, Hallway1, Hallway2, Hallway3

#########################################
#               Types:
#########################################

# States & Cells


const Location = SVector{2, Int}

const Orientation = Int
const NORTH :: Orientation = 0; const EAST :: Orientation = 1
const SOUTH :: Orientation = 2; const WEST :: Orientation = 3
struct Position
    location::Location
    orientation::Orientation
end

const CellType = Int
const Empty :: CellType = 0
const Wall :: CellType =  1
const Hole :: CellType = 2
const Mark :: CellType = 3

abstract type HoleEffect end
struct Terminate <: HoleEffect end
struct Reset <: HoleEffect end

const Side = Int
const FRONT :: Side = 0; const RIGHT :: Side = 1
const BACK :: Side = 2; const LEFT :: Side = 3

# Actions & Transitions

const Action = Int
const GoRight :: Action = 1; const GoBack :: Action = 2; const GoLeft :: Action = 3
const GoForward :: Action = 4; const NoOp :: Action = 5; const Measure :: Action = 6

abstract type ActionEffect end
struct Normal <: ActionEffect end
struct DoubleAction <: ActionEffect end
struct Deviate <: ActionEffect end
struct NoAction <: ActionEffect end
struct CustomActionEffect <: ActionEffect
    movement::Dict{Action, Any}
end

Base.@kwdef struct Transition_Type
    probability::Float64        = 1
    effect::ActionEffect       = Deviate 
end

# Observations

abstract type Observations end
struct EmptyObs <: Observations end
struct NeighbourObs <: Observations end
struct CustomRelState <: Observations
    obs::Dict{CellType, Any}
end

abstract type ObservedInfo end
struct FullState <: ObservedInfo end
struct RelState <: ObservedInfo end

Base.@kwdef struct Observation_Type
    observedinfo::ObservedInfo  = FullState()
    probability::Float64        = 1
    effect::Observations        = EmptyObs()
end

struct Relative_Observation
    front::CellType
    right::CellType
    back::CellType
    left::CellType
end
Relative_Observation(v::Vector{CellType}) = Relative_Observation(v...)

# Envs

abstract type GridWorld{S, A, O} <: POMDP{S,A,O} end

Base.@kwdef struct LocationGridWorld <: GridWorld{Location, Action, Location}
    size::Tuple{Int, Int}               = (10,10)
    observation_type::Observation_Type  = Observation_Type()
    transition_type::Transition_Type    = Transition_Type()
    rewards::Dict{Location, Float64}    = Dict(Location(10,10) => 1)
    holes::Set{Location}                = Set(keys(rewards))
    walls::Set{Location}                = Set()
    marks::Set{Location}                = Set()
    hole_effect:: HoleEffect            = Terminate()
    discount::Float64                   = 0.95
    initial_state::Any                  = Deterministic(Location(0,0))
end

Base.@kwdef struct PositionGridWorld <: GridWorld{Position, Action, Relative_Observation}
    size::Tuple{Int, Int}               = (10,10)
    observation_type::Observation_Type  = Observation_Type()
    transition_type::Transition_Type    = Transition_Type()
    rewards::Dict{Location, Float64}    = Dict(Location(10,10) => 1)
    holes::Set{Location}                = Set(keys(rewards))
    walls::Set{Location}                = Set()
    marks::Set{Location}                = Set()
    hole_effect:: HoleEffect            = Terminate()
    discount::Float64                   = 0.95
    initial_state::Any                  = Deterministic(Location(0,0))
end

#########################################
#               Defaults:
#########################################

FrozenLakeSmall = LocationGridWorld(
    (4,4),
    Observation_Type(FullState(), 0.5, EmptyObs()),
    Transition_Type(0.5, DoubleAction()),
    Dict(Location(4,4)=>1),
    Set([Location(2,2), Location(1,4), Location(4,2), Location(4,3), Location(4,4)]),
    Set([]),
    Set([]),
    Terminate(),
    0.99,
    Deterministic(Location(1,1))
    )

FrozenLakeLarge = LocationGridWorld(
    (10,10),
    Observation_Type(FullState(), 0.5, EmptyObs()),
    Transition_Type(0.5, DoubleAction()),
    Dict(Location(10,10)=>1),
    Set([   Location(1,4), Location(2,4), Location(3,7), Location(5,4), Location(5,8),
            Location(6,2), Location(6,5), Location(7,1), Location(7,3), Location(8,9),
            Location(9,3), Location(9,6), Location(10,8), Location(10,10)]),
    Set([]),
    Set([]),
    Terminate(),
    0.99,
    Deterministic(Location(1,1))
    )

movement_hallway = Dict{Action, Any}(
    GoForward   => SparseCat([  [GoForward], [NoOp], [GoLeft, GoForward], [GoRight, GoForward], 
                                [GoBack, GoForward], [GoBack, GoForward, GoBack] ], 
                                    [0.8, 0.05, 0.05, 0.05, 0.025, 0.025 ]  ),
    GoLeft      => SparseCat([ [GoLeft], [NoOp], [GoRight], [GoBack] ], [0.7, 0.1, 0.1, 0.1]),
    GoRight     => SparseCat([ [GoRight], [NoOp], [GoLeft], [GoBack] ], [0.7, 0.1, 0.1, 0.1]),
    GoBack      => SparseCat([ [GoBack], [NoOp], [GoLeft], [GoRight] ], [0.6, 0.1, 0.15, 0.15]),
    NoOp        => Deterministic([NoOp]),
    Measure      => Deterministic([NoOp])
)

obs_hallway     = Dict{CellType, Any}(
    Empty   => SparseCat([Empty, Wall], [0.95, 0.05]),
    # Empty => Deterministic(Empty),
    Wall    => SparseCat([Wall, Empty], [0.9, 0.1]),
    # Wall => Deterministic(Wall),
    # This is slightly different from the original implementation, where marks are only observable when facing them.
    # Thats really annoying to implement, though, so we won't.
    Mark    => Deterministic(Mark), 
    # No holes exist here.
    Hole    => SparseCat([Empty, Wall], [0.95, 0.05]),
    # Hole => Deterministic(Hole)
)

CustomMiniHallway = PositionGridWorld(
    (2,3),
    # Observation_Type(RelState(), 0.0, CustomRelState(obs_hallway)),
    Observation_Type(RelState(), 1, EmptyObs()),
    Transition_Type(0.0, CustomActionEffect(movement_hallway)),
    Dict(Location(2,3) => 1),
    Set([Location(2,3)]),
    Set([Location(1,3), Location(2,1)]),
    Set([]),
    Reset(),
    0.99,
    Deterministic(Position(Location(1,1), EAST))
)

Hallway1 = PositionGridWorld(
    (11,2),
    Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    Dict(Location(9,2) => 1),
    Set([Location(9,2)]),
    Set([Location(1,2), Location(2,2), Location(4,4), Location(6,2), Location(8,2), Location(10,2), Location(11,2)]),
    Set([Location(3,2), Location(5,2), Location(7,2)]),
    Reset(),
    0.99,
    Deterministic(Position(Location(1,1), EAST)) # TODO: should be Uniform()
)

Hallway2 = PositionGridWorld(
    (7,5),
    Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    Dict(Location(7,4) => 1),
    Set([Location(7,4)]),
    Set([Location(1,1), Location(1,3), Location(1,5), Location(3,2), Location(3,3), Location(3,4),
         Location(5,2), Location(5,3), Location(5,4), Location(5,1), Location(5,3), Location(5,5)]),
    Set([]),
    Reset(),
    0.99,
    Deterministic(Position(Location(1,1), EAST)) # TODO: should be Uniform()
)

Hallway3 = PositionGridWorld(
    (5,2),
    Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    Dict(Location(3,1) => 1, Location(1,1) => -1, Location(5,1) => -1),
    Set([Location(3,1), Location(1,1), Location(5,1)]),
    Set([Location(3,2)]),
    Set([]),
    Reset(),
    0.99,
    SparseCat(Position(Location(2,2), NORTH), Position(Location(4,2), NORTH))
)


#########################################
#      Helper Logic Functions:
#########################################

function get_location_type(model::GridWorld, s::Location)
    !(is_on_grid(model,s)) && return Wall
    s in model.walls && return Wall
    s in model.holes && return Hole
    s in model.marks && return Mark
    return Empty
end

is_on_grid(model::GridWorld, s::Location) = (1 <= s[1] <= model.size[1]) && (1 <= s[2] <= model.size[2])

rel_position = Dict{Orientation, Dict{Side, Location}}(
    NORTH => Dict{Side, Location}( FRONT => Location(0,-1), BACK => Location(0,1), LEFT => (-1,0), RIGHT => (1,0) ),
    SOUTH => Dict{Side, Location}( FRONT => Location(0,1), BACK => Location(0,-1), LEFT => (1,0), RIGHT => (-1,0) ),
    EAST  => Dict{Side, Location}( FRONT => Location(1,0), BACK => Location(-1,0), LEFT => (1,0), RIGHT => (-1,0) ),
    WEST  => Dict{Side, Location}( FRONT => Location(-1,0), BACK => Location(1,0), LEFT => (-1,0), RIGHT => (1,0) )
)

action_to_Orientation = Dict(GoForward => NORTH, GoLeft => EAST, GoBack => BACK, GoRight => WEST)

get_relative_location(s::Position, side::Side) = s.location + rel_position[s.orientation][side]

#########################################
#      States, Obs, Actions:
#########################################

function NullObs(m::X) where X<:GridWorld
    obsinfo = m.observation_type.observedinfo
    if obsinfo == FullState()
        return Location(-1,-1)
    elseif obsinfo == RelState
        return Relative_Observation(Wall, Wall, Wall, Wall)
    end
end

function SinkState(m::X) where X<:GridWorld
    statetype = POMDPs.statetype(m)
    statetype == Location && (return Location(-1,-1))
    statetype == Position && (return Position(Location(-1,-1), NORTH))
end

function POMDPs.states(m::X) where X<:GridWorld
    statetype = POMDPs.statetype(m)
    if statetype == Location
        ss = vec(Location[Location(x,y) for x in 1:m.size[1], y in 1:m.size[2]])
        push!(ss, SinkState(m))
        return ss
    elseif statetype == Position 
        locations = vec(Location[Location(x,y) for x in 1:m.size[1], y in 1:m.size[2]])
        orientations :: Vector{Orientation} = [NORTH, SOUTH, EAST, WEST]
        prod = vec(collect(Iterators.product(locations, orientations)))
        ss = []
        for (loc, or) in prod
            push!(ss, Position(loc,or))
        end
        # TODO: Figure out why this does not work !!!
        # ss = map( (loc, or) -> (Position(loc,or)),  prod )
        push!(ss, SinkState(m))
        return ss
    end
end

POMDPs.initialstate(m::X) where X<:GridWorld = m.initial_state
POMDPs.isterminal(m::X, s) where X <: GridWorld = s == SinkState(m)

function POMDPs.stateindex(m::X, s::Location) where X<:GridWorld
    s == SinkState(m) && return length(states(m))
    return s[1] + m.size[1] * (s[2]-1)
end

function POMDPs.stateindex(m::X, s::Position) where X<:GridWorld
    s == SinkState(m) && return length(states(m))
    return s.location[1] + m.size[1] * ( (s.location[2]-1) + m.size[2] * s.orientation)
end

POMDPs.actions(m::X) where X<:GridWorld = [GoForward, GoLeft, GoBack, GoRight, NoOp]
POMDPs.actionindex(mdp::X, a) where X<:GridWorld = a

function allCombs(A, n::Int)
    """Return all possible tuples of lenght n with elements from A"""
    list = []
    for el in Iterators.product(fill(A,n)...)
        push!(list, el)
    end
    return list
end

all_rel_observations = map( s -> Relative_Observation(s[1], s[2], s[3], s[4]),  
                            allCombs([Empty, Wall, Hole, Mark], 4) )
                            # allCombs([Empty, Wall], 4) )
# println(Iterators.product(ntuple(i->[Empty, Wall, Hole, Mark], 4)...))
rel_obs_idxs = Dict(zip(all_rel_observations, 1:length(all_rel_observations)))

function POMDPs.observations(m::X) where X<:GridWorld
    obsinfo = m.observation_type.observedinfo
    if obsinfo isa FullState
        obs = POMDPs.states(m)
        return obs
    elseif obsinfo isa RelState || obsinfo isa CustomRelState
        return all_rel_observations
    end
end

function POMDPs.obsindex(m::X, o) where X<:GridWorld
    if m.observation_type.observedinfo == FullState()
        o != NullObs(m) ? (return stateindex(m, o)) : (return prod(m.size) + 1)
    elseif m.observation_type.observedinfo == RelState()
        return rel_obs_idxs[o]
    end
end

#########################################
#      Transitions:
#########################################

function transition_normal(m::X, s::Location, a::Action) where X <: GridWorld
    a in [NoOp, Measure] && return s
    or = action_to_Orientation[a]
    s_next = get_relative_location(Position(s,or), FRONT)
    get_location_type(m,s_next) == Wall ? (return s) : (return s_next)
end

function transition_normal(m::X, s::Position, a::Action) where X <: GridWorld
    # Moving
    a in [NoOp, Measure] && return s
    if a == GoForward
        rel_loc = get_relative_location(s, FRONT)
        get_location_type(m,rel_loc) == Wall ? (return s) : (return Position(rel_loc, s.orientation))
    # Turning
    else
        orientation = Orientation( ( Int(s.orientation) + Int(a) )  % 4 )
        return Position(s.location, orientation)
    end
end

@memoize LRU(maxsize=10_000) function POMDPs.transition(m::X, s, a) where X<:GridWorld
    s isa Location ? loc = s : loc = s.location
    if loc in m.holes || s == SinkState(m)
        m.hole_effect == Terminate() && return Deterministic(SinkState(m))
        m.hole_effect == Reset() && return m.initial_state
    end

    T = m.transition_type

    s_normal = transition_normal(m, s, a)
    p = T.probability

    if T.effect == Normal() || p == 1
        return Deterministic(s_normal)
    elseif T.effect == DoubleAction()
        s_double = transition_normal(m, s_normal, a)
        return SparseCat([s_normal, s_double], [p, 1-p])
    elseif T.effect == Deviate()
        printdb("Not Implemented")
        return Deterministic(s_normal)
    elseif T.effect == NoAction()
        return SparseCat([s, s_normal], [1-p, p])


    elseif T.effect isa CustomActionEffect
        outcomes = Dict()
        for (actions, prob) in weighted_iterator(T.effect.movement[a])
            sp = s
            for ap in actions
                sp = transition_normal(m,sp,ap)
            end
            haskey(outcomes, sp) ? outcomes[sp] += prob : outcomes[sp] = prob
        end
        return SparseCat(collect(keys(outcomes)), collect(values(outcomes)))
    end
    println("Error: transition type not implemented!")
    return Deterministic(s)
end

POMDPs.isterminal(m::X, s) where X<:GridWorld = s == SinkState(m)

#########################################
#      Observations:
#########################################

@memoize LRU(maxsize=10_000) function observation_normal(m::X, a, sp) where X<:GridWorld
    O = m.observation_type

    if O.observedinfo == FullState()
        return sp
    elseif O.observedinfo == RelState()
        f_obs, r_obs, b_obs, l_obs = map( side -> get_location_type(m, get_relative_location(sp,side)), [FRONT, RIGHT, BACK, LEFT]  )
        return Relative_Observation( f_obs, r_obs, b_obs, l_obs) 
    end
end

@memoize LRU(maxsize=10_000) function get_custom_obs(m::X, o_normal) where X <: GridWorld
    obs_vec = [o_normal.front, o_normal.right, o_normal.back, o_normal.left]
    univariate_outcomes = Vector{Tuple{CellType, Float64}}[]
    for obs in obs_vec
        this_outcomes = Tuple{CellType, Float64}[]
        o_distr = m.observation_type.effect.obs[obs]
        for (o, p) in weighted_iterator(o_distr)
            push!(this_outcomes, (o,p))
        end
        push!(univariate_outcomes, this_outcomes)
    end
    product = custom_discrete_product(univariate_outcomes; mapfunction=Relative_Observation)
    return product
end



function POMDPs.observation(m::X, a, sp) where X<:GridWorld
    O = m.observation_type

    o_normal = observation_normal(m,a,sp)
    p = O.probability

    if p == 1 && !(O.effect isa CustomRelState)
        return Deterministic(o_normal)
    elseif O.effect == EmptyObs()
        return SparseCat([o_normal, NullObs(m)], [p, 1-p])
    elseif O.effect == NeighbourObs()
        printdb("Error: observatin type NeighbourObs not yet implemented!")
        return o_normal

    elseif O.effect isa CustomRelState
        return get_custom_obs(m, o_normal)
    end
    print("Help!")
    return Deterministic(o_normal)
end

# Rewards

POMDPs.reward(m::X, s::Location, a) where X<:GridWorld = get(m.rewards, s, 0.0) 
POMDPs.reward(m::X, s::Position, a) where X<:GridWorld = get(m.rewards, s.location, 0.0)

# discount

POMDPs.discount(mdp::X)where X<:GridWorld = mdp.discount

# Conversion
# function POMDPs.convert_a(::Type{V}, a::Symbol, m::GridWorld) where {V<:AbstractArray}
#     convert(V, [aind[a]])
# end
# function POMDPs.convert_a(::Type{Symbol}, vec::V, m::GridWorld) where {V<:AbstractArray}
#     actions(m)[convert(Int, first(vec))]
# end


#########################################
#           Helping Functions:
#########################################

function custom_discrete_product(A::Vector{Vector{Tuple{X, Float64}}}; mapfunction=Tuple) where X<:Any
    """Creates a Sparse Univariate distribution over factorized states.
    Input: A vector with, for each Factor, a vector with tuples of their possible value and a probability.
    """
    # Initialize: place all elements from the first factor in the new list
    outcomes, probs = Vector{Vector{X}}(), Vector{Float64}()
    for (ind,p) in A[1]
        push!(outcomes, [ind])# Initialize: place all elements from the first factor in the new list
        push!(probs, p)
    end
    # For each factor, create a combination of each outcome and each previous outcome and place these in newTuples
    for (i,thisFactor) in enumerate(A[2:length(A)])
        newoutcomes, newprobs = Vector{Vector{X}}(), Vector{Float64}()
        for (val, p_val) in thisFactor
            for (lst, p_lst) in zip(outcomes, probs)
                push!(newoutcomes, push!(deepcopy(lst), val))
                push!(newprobs, p_lst*p_val)
            end
        end
        outcomes, probs = newoutcomes, newprobs
    end
    # Return a distribution
    outcomes = map(mapfunction, outcomes)
    return SparseCat(outcomes, probs)
end

end