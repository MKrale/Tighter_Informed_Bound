#Copied gridworld definition from POMDPModels, but now as a POMDP.

module CustomGridWorlds
using POMDPs
using POMDPTools
using Distributions
using StaticArrays
using Random
using Memoization, LRUCache
using OrderedCollections


export Gridworld, FrozenLakeSmall, FrozenLakeLarge, CustomMiniHallway, Hallway1, Hallway2, TigerGrid

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
struct Sink <: HoleEffect end

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
struct TrueObs <: Observations end
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

Base.@kwdef mutable struct LocationGridWorld <: GridWorld{Location, Action, Location}
    size::Tuple{Int, Int}               = (10,10)
    observation_type::Observation_Type  = Observation_Type()
    transition_type::Transition_Type    = Transition_Type()
    rewards::Dict{Location, Float64}    = Dict(Location(10,10) => 1)
    holes::Set{Location}                = Set(keys(rewards))
    walls::Set{Location}                = Set()
    marks::Set{Location}                = Set()
    hole_effect:: HoleEffect            = Reset()
    discount::Float64                   = 0.95
    initial_state::Any                  = Deterministic(Location(0,0))
    measuring_effect::Observation_Type  = Observation_Type(FullState(), 0.0, EmptyObs())
end

Base.@kwdef mutable struct PositionGridWorld <: GridWorld{Position, Action, Relative_Observation}
    size::Tuple{Int, Int}               = (10,10)
    observation_type::Observation_Type  = Observation_Type()
    transition_type::Transition_Type    = Transition_Type()
    rewards::Dict{Location, Float64}    = Dict(Location(10,10) => 1)
    holes::Set{Location}                = Set(keys(rewards))
    walls::Set{Location}                = Set()
    marks::Set{Location}                = Set()
    hole_effect:: HoleEffect            = Reset()
    discount::Float64                   = 0.95
    initial_state::Any                  = Deterministic(Location(0,0))
    measuring_effect::Observation_Type  = Observation_Type(FullState(), 1, EmptyObs())
end


#########################################
#               Defaults:
#########################################

FrozenLakeSmall = LocationGridWorld(
    size                = (4,4),
    observation_type    = Observation_Type(FullState(), 0.5, EmptyObs()),
    transition_type     = Transition_Type(0.5, DoubleAction()),
    rewards             = Dict(Location(4,4)=>1),
    holes               = Set([Location(2,2), Location(1,4), Location(4,2), Location(4,3), Location(4,4)]),
    walls               = Set([]),
    marks               = Set([]),
    hole_effect         = Terminate(),
    discount            = 0.95,
    initial_state       = Deterministic(Location(1,1))
    )

FrozenLakeLarge = LocationGridWorld(
    size                = (10,10),
    observation_type    = Observation_Type(FullState(), 0.5, EmptyObs()),
    transition_type     = Transition_Type(0.5, DoubleAction()),
    rewards             = Dict(Location(10,10)=>1),
    holes               = Set([   Location(1,4), Location(2,4), Location(3,7), Location(5,4), Location(5,8),
            Location(6,2), Location(6,5), Location(7,1), Location(7,3), Location(8,9),
            Location(9,3), Location(9,6), Location(10,8), Location(10,10)]),
    walls               = Set([]),
    marks               = Set([]),
    hole_effect         = Terminate(),
    discount            = 0.95,
    initial_state       = Deterministic(Location(1,1))
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
    # Hole    => SparseCat([Empty, Wall], [0.95, 0.05]),
    Hole => SparseCat([Empty, Wall], [0.95, 0.05]),
    # Hole => Deterministic(Hole)
)

CustomMiniHallway = PositionGridWorld(
    size                = (2,3),
    observation_type    = Observation_Type(RelState(), 0.0, CustomRelState(obs_hallway)),
    # observation_type    = Observation_Type(RelState(), 1, EmptyObs()),
    transition_type     = Transition_Type(0.0, CustomActionEffect(movement_hallway)),
    rewards             = Dict(Location(2,3) => 1),
    holes               = Set([Location(2,3)]),
    walls               = Set([Location(1,3), Location(2,1)]),
    marks               = Set([]),
    hole_effect         = Reset(),
    discount            = 0.95,
    initial_state       = Deterministic(Position(Location(1,1), EAST))
)

Hallway1 = PositionGridWorld(
    size                = (11,2),
    observation_type    = Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    # observation_type    = Observation_Type(RelState(), 1, EmptyObs()),
    transition_type     = Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    rewards             = Dict(Location(9,2) => 1),
    holes               = Set([Location(9,2)]),
    walls               = Set([Location(1,2), Location(2,2), Location(4,2), Location(6,2), Location(8,2), Location(10,2), Location(11,2)]),
    marks               = Set([Location(3,2), Location(5,2), Location(7,2)]),
    hole_effect         = Reset(),
    discount            = 0.95,
    initial_state       = :Uniform
    # initial_state       = Deterministic(Position(Location(1,1), EAST)) # TODO: should be Uniform()
)

Hallway2 = PositionGridWorld(
    size                = (7,5),
    observation_type    = Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    # observation_type    = Observation_Type(RelState(), 1, EmptyObs()),
    transition_type     = Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    rewards             = Dict(Location(7,4) => 1),
    holes               = Set([Location(7,4)]),
    walls               = Set([Location(1,1), Location(1,3), Location(1,5), Location(3,2), Location(3,3), Location(3,4),
                                Location(5,2), Location(5,3), Location(5,4), Location(7,1), Location(7,3), Location(7,5)]),
    marks               = Set([]),
    hole_effect         = Reset(),
    discount            = 0.95,
    initial_state       = :Uniform
    # initial_state       = Deterministic(Position(Location(1,2), EAST)) # TODO: should be Uniform()
)

TigerGrid = PositionGridWorld(
    size                = (5,2),
    observation_type    = Observation_Type(RelState(), 1, CustomRelState(obs_hallway)),
    # observation_type    = Observation_Type(RelState(), 1, EmptyObs()),
    transition_type     = Transition_Type(0.8, CustomActionEffect(movement_hallway)),
    rewards             = Dict(Location(3,1) => 1, Location(1,1) => -1, Location(5,1) => -1),
    holes               = Set([Location(3,1), Location(1,1), Location(5,1)]),
    walls               = Set([Location(3,2)]),
    marks               = Set([]),
    hole_effect         = Reset(),
    discount            = 0.95,
    initial_state       = SparseCat([Position(Location(2,2), NORTH), Position(Location(4,2), NORTH)], [0.5, 0.5])
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

states_idxs_dict = LRU(maxsize=10)
function states_and_idxs(m::X) where X<:GridWorld
    m_id = objectid(m)
    haskey(states_idxs_dict, m_id) && return states_idxs_dict[m_id]
    statetype = POMDPs.statetype(m)
    if statetype == Location
        ss = vec(Location[Location(x,y) for x in 1:m.size[1], y in 1:m.size[2]])
        ss = ss[findall(s -> !(get_location_type(m, s) in [Wall]), ss)]
        push!(ss, SinkState(m))
        dict =  OrderedDict(zip(ss, 1:length(ss)))
    elseif statetype == Position 
        locations = vec(Location[Location(x,y) for x in 1:m.size[1], y in 1:m.size[2]])
        orientations :: Vector{Orientation} = [NORTH, SOUTH, EAST, WEST]
        prod = vec(collect(Iterators.product(locations, orientations)))
        ss = []
        for (loc, or) in prod
            push!(ss, Position(loc,or))
        end
        ss = ss[findall(s -> !(get_location_type(m, s.location) in [Wall]), ss)]
        # TODO: Figure out why this does not work !!!
        # ss = map( (loc, or) -> (Position(loc,or)),  prod )
        push!(ss, SinkState(m))
        dict = OrderedDict(zip(ss, 1:length(ss)))
    end
    states_idxs_dict[m_id] = dict
    return dict 
end

POMDPs.states(m::X) where X<:GridWorld = collect(keys(states_and_idxs(m)))
POMDPTools.ordered_states(m::X) where X<:GridWorld = states(m)
POMDPs.stateindex(m::X, s) where X<:GridWorld = states_and_idxs(m)[s]

init_state_dict = LRU(maxsize=10)
function POMDPs.initialstate(m::X) where X<:GridWorld
    m_id = objectid(m)
    haskey(init_state_dict, m_id) && return init_state_dict[m_id]
    m.initial_state == :Uniform ? (sinit = uniform_state_dist(m)) : (sinit = m.initial_state)
    init_state_dict[m_id] = sinit 
    return sinit
end
function uniform_state_dist(m::X) where X <: GridWorld
    ss = states(m)[findall(s -> s != SinkState(m), states(m))]
    p = 1.0/length(ss)
    return SparseCat(ss, repeat([p], length(ss)))
end

POMDPs.isterminal(m::X, s) where X <: GridWorld = m.hole_effect == Sink() ? false : s == SinkState(m)

# function POMDPs.stateindex(m::X, s::Location) where X<:GridWorld
#     s == SinkState(m) && return length(states(m))
#     return s[1] + m.size[1] * (s[2]-1)
# end

# function POMDPs.stateindex(m::X, s::Position) where X<:GridWorld
#     s == SinkState(m) && return length(states(m))
#     return s.location[1] + m.size[1] * ( (s.location[2]-1) + m.size[2] * s.orientation)
# end

function POMDPs.actions(m::X) where X<:GridWorld
    if m.measuring_effect.effect isa EmptyObs
        return [GoRight, GoBack, GoLeft, GoForward, NoOp]
    end
    return [GoRight, GoBack, GoLeft, GoForward, NoOp, Measure]
end
POMDPTools.ordered_actions(m::X) where X <: GridWorld = POMDPs.actions(m)
POMDPs.actionindex(mdp::X, a) where X<:GridWorld = a

obs_idxs_dict = LRU(maxsize=10)
function obs_and_idxs(m::X) where X<:GridWorld
    m_id = objectid(m)
    haskey(obs_idxs_dict, m_id) && return obs_idxs_dict[m_id]

    Os = Set()    
    for s in states(m)
        for a in actions(m)
            for o in support(observation(m,a,s))
                push!(Os, o)
            end
        end
    end
    push!(Os, NullObs(m))
    dict = OrderedDict(zip(Os, 1:length(Os)))
    obs_idxs_dict[m_id] = dict 
    return dict
end


POMDPs.observations(m::X) where X<:GridWorld = collect(keys(obs_and_idxs(m)))
POMDPTools.ordered_observations(m::X) where X<: GridWorld = observations(m)
POMDPs.obsindex(m::X,o) where X<:GridWorld = obs_and_idxs(m)[o]

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

trans_dict = LRU(maxsize=10_000)
function POMDPs.transition(m::X, s, a) where X<:GridWorld
    m_id = objectid(m)
    haskey(trans_dict, (m_id,s,a)) && return trans_dict[(m_id,s,a)]

    s isa Location ? loc = s : loc = s.location
    T = m.transition_type
    s_normal = transition_normal(m, s, a)
    p = T.probability
    snext = nothing
    if loc in m.holes || s == SinkState(m)
        m.hole_effect in [Terminate(), Sink()] && (snext = Deterministic(SinkState(m)))
        m.hole_effect == Reset() && (snext = initialstate(m))
    elseif T.effect == Normal() || p == 1
        snext = Deterministic(s_normal)
    elseif T.effect == DoubleAction()
        s_double = transition_normal(m, s_normal, a)
        snext = SparseCat([s_normal, s_double], [p, 1-p])
    elseif T.effect == Deviate()
        printdb("Not Implemented")
        snext = Deterministic(s_normal)
    elseif T.effect == NoAction()
        snext = SparseCat([s, s_normal], [1-p, p])


    elseif T.effect isa CustomActionEffect
        outcomes = Dict()
        for (actions, prob) in weighted_iterator(T.effect.movement[a])
            sp = s
            for ap in actions
                sp = transition_normal(m,sp,ap)
            end
            haskey(outcomes, sp) ? outcomes[sp] += prob : outcomes[sp] = prob
        end
        snext = SparseCat(collect(keys(outcomes)), collect(values(outcomes)))
    else
        println("Error: transition type not implemented!")
        snext = Deterministic(s)
    end
    trans_dict[(m_id,s,a)] = snext 
    return snext
end

# POMDPs.isterminal(m::X, s) where X<:GridWorld = m.hole_effect == Sink() ? false : s == SinkState(m)

#########################################
#      Observations:
#########################################

function observation_normal(m::X, a, sp) where X<:GridWorld
    O = m.observation_type

    if O.observedinfo == FullState()
        return sp
    elseif O.observedinfo == RelState()
        f_obs, r_obs, b_obs, l_obs = map( side -> get_location_type(m, get_relative_location(sp,side)), [FRONT, RIGHT, BACK, LEFT]  )
        return Relative_Observation( f_obs, r_obs, b_obs, l_obs) 
    end
end

function get_custom_obs(m::X, o_normal) where X <: GridWorld
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


obs_dict = LRU(maxsize=10_000)
function POMDPs.observation(m::X, a, sp) where X<:GridWorld
    m_id = objectid(m)
    haskey(obs_dict, (m_id,a,sp)) && return obs_dict[(m_id,a,sp)]
    sp == SinkState(m) && return Deterministic(NullObs(m))
    O = m.observation_type

    o_normal = observation_normal(m,a,sp)
    p = O.probability

    obs = nothing
    if p == 1 && !(O.effect isa CustomRelState)
        obs = Deterministic(o_normal)
    elseif O.effect == EmptyObs()
        obs = SparseCat([o_normal, NullObs(m)], [p, 1-p])
    elseif O.effect == NeighbourObs()
        printdb("Error: observatin type NeighbourObs not yet implemented!")
        obs = o_normal

    elseif O.effect isa CustomRelState
        obs = get_custom_obs(m, o_normal)
    else
        print("Oh-oh!")
        obs = Deterministic(o_normal)
    end
    obs_dict[(m_id,a,sp)] = obs
    return obs
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
