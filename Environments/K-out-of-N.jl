# Imports

module K_out_of_Ns

using Base.Iterators, POMDPs, QuickPOMDPs, POMDPTools, Distributions

export K_out_of_N

################################
#       Model Parameters
################################

# Parameters as used in other functions.
@kwdef mutable struct K_out_of_N <: POMDP{Tuple{Vararg{Int}},Tuple{Vararg{Int}},Tuple{Vararg{Int}}}
    N::Integer                  = 3
    K::Integer                  = 3
    smax::Integer               = 3
    spread:: Integer            = 2
    p0::Float64                 = 0.2
    p1::Float64                 = 0.5
    p2::Float64                 = 0.9
    break_cost::Float64         = 0.5
    repair_cost::Float64        = 0.25
    inspect_cost::Float64       = 0.05
    deterministic_obs::Bool     = true
    discount::Float64           = 0.95
end

# default_args = Dict{Symbol, Any}(
#     :N=>3, :K=>3, :smax=>3,
#     :spread=>2, :p0=>0.2, :p1=>0.5, :p2=>0.9,
#     :break_cost=>0.5, :repair_cost=>0.25, :inspect_cost=>0.05, :deterministic_obs=>true )

# function K_out_of_N(D::Dict{Symbol,Any})
#     for key in keys(default_args)
#         haskey(D,key) || (D[key] = default_args[key])
#     end
#     return K_out_of_N(D[:N], D[:K], D[:smax], 
#             D[:spread], D[:p0], D[:p1], D[:p2], 
#             D[:break_cost], D[:repair_cost], D[:inspect_cost], 
#             D[:deterministic_obs])
# end

# K_out_of_N() = K_out_of_N(default_args...)
# K_out_of_N(N::Int, K::Int; deterministic_obs=true) = K_out_of_N(Dict{Symbol, Any}(:N=>N, :K=>K, :deterministic_obs=>deterministic_obs))


################################
#       Helper Functions:
################################

# Listing all possible combinations of factorized states (defunct)
function allCombs(A, n)
    """Return all possible tuples of lenght n with elements from A"""
    list = []
    for el in product(fill(A,n)...)
        push!(list, el)
    end
    return list
end

function custom_discrete_product(A::Vector{Vector{Tuple{Integer, AbstractFloat}}})
    """Creates a Sparse Univariate distribution over factorized states.
    Input: A vector with, for each Factor, a vector with tuples of their possible value and a probability.
    """
    # Initialize: place all elements from the first factor in the new list
    outcomes, probs = Vector{Vector{Integer}}(), Vector{AbstractFloat}()
    for (ind,p) in A[1]
        push!(outcomes, [ind])# Initialize: place all elements from the first factor in the new list
        push!(probs, p)
    end
    # For each factor, create a combination of each outcome and each previous outcome and place these in newTuples
    for (i,thisFactor) in enumerate(A[2:length(A)])
        newoutcomes, newprobs = Vector{Vector{Integer}}(), Vector{AbstractFloat}()
        for (val, p_val) in thisFactor
            for (lst, p_lst) in zip(outcomes, probs)
                push!(newoutcomes, push!(deepcopy(lst), val))
                push!(newprobs, p_lst*p_val)
            end
        end
        outcomes, probs = newoutcomes, newprobs
    end
    # Return a distribution
    outcomes = map(Tuple, outcomes)
    return SparseCat(outcomes, probs)
end

function defactorize(x, lengths::Tuple{Int,Int})
    length(lengths)==1 && return x
    factor = 1
    low, high = lengths
    val = 1 # Thanks Julia!
    for el in x
        val += (el-low)*factor
        factor = factor * (high-low+1)
    end
    return Integer(val)
end


################################
#       Model Functions:
################################

T = function(M::K_out_of_N,s,a)
    """Transition function"""
    s_extended = append!( [0], push!(collect(deepcopy(s)), 0))
    p = Vector{Vector{Tuple{Integer, AbstractFloat}}}()
    for i=range(1,M.N)
        if a[i] == 1
            push!(p, [(0,1.0)])
        elseif s[i] < M.smax
            broken_neighbours = Int(s_extended[i] == M.smax) + Int(s_extended[i+2] == M.smax)
            if broken_neighbours == 0
                thisp = M.p0
            elseif broken_neighbours == 1
                thisp = M.p1
            elseif broken_neighbours == 2
                thisp = M.p2
            end
            push!(p, [(s[i],1-thisp), (s[i]+1, thisp)])
        elseif s[i] == M.smax
            push!(p, [(s[i],1.0)])
        end
    end
    return custom_discrete_product(p)
end

O = function(M::K_out_of_N,s,a)
    """Observation function"""
    obs = ones(M.N)
    for i=1:M.N
        if a[i] == 1 || a[i] == 2
            obs[i] = s[i]
        end
    end
    return Deterministic(Tuple(obs))
end

R = function(M::K_out_of_N,s,a)
    """Reward Funtion"""
    r = 0
    for i=range(1,M.N)
        if s[i]==M.smax
            r -= M.break_cost
        end
        if a[i] == 1
            r -= M.repair_cost
        elseif a[i] == 2
            r -= M.inspect_cost
        end
    end
    return r #+ r_plus # (M.break_cost + M.repair_cost + M.inspect_cost) * M.N
end

r_plus = 1
################################
#       Model Definition:
################################

POMDPs.states(M::K_out_of_N) = allCombs(0:M.smax , M.N)
POMDPs.actions(M::K_out_of_N) = allCombs(0:2,M.N)
POMDPs.observations(M::K_out_of_N) = allCombs(0:M.smax , M.N)
POMDPs.transition(M::K_out_of_N, s,a) = T(M,s,a)
POMDPs.observation(M::K_out_of_N, a,sp) = O(M,sp,a)
POMDPs.reward(M::K_out_of_N, s,a) = R(M,s,a)
POMDPs.discount(M::K_out_of_N) = M.discount
POMDPs.initialstate(M::K_out_of_N) = Deterministic(Tuple(fill(1,M.N)))

POMDPs.actiontype(M::K_out_of_N) = NTuple{M.N, Integer}
POMDPs.statetype(M::K_out_of_N) = NTuple{M.N, Integer}
POMDPs.obstype(M::K_out_of_N) = NTuple{M.N, Integer}

POMDPs.stateindex(M::K_out_of_N, s) = defactorize(s, (0,M.smax))
POMDPs.actionindex(M::K_out_of_N, a) = defactorize(a, (0,2))
POMDPs.obsindex(M::K_out_of_N, o) = defactorize(o, (0,M.smax))

end
