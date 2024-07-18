# tolerance is ||alpha^k - alpha^k+1||_infty


#########################################
#               Solver:
#########################################

struct BIBSolver <: Solver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end

function BIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3)
    return BIBSolver(max_iterations, precision)
end

struct BIB_Data
    Q::Array{Float64,2}
    B_idx::Array{Int,3}
    P_obs::Array{Float64,3}
    Obs::Array{Vector{Int},2}
end

function POMDPs.solve(solver::BIBSolver, model::POMDP)

    S, A, O = states(model), actions(model), observations(model) #TODO: figure out if I need to use ordered_... here
    S_dict, A_dict, O_dict = Dict(zip(S, 1:length(S))), Dict(zip(A, 1:length(A))), Dict(zip(O, 1:length(O)))
    ns, na, no = length(states(model)), length(actions(model)), length(observations(model))
    # Bs = Array{DiscreteBelief,3}(undef, ns,na,no)
    
    B_idx = zeros(Int,ns,na,no)
    B = Array{DiscreteHashedBelief,1}()
    U = DiscreteHashedBeliefUpdater(model)

    # 1: Precompute observation probabilities
    P_obs = zeros(no,ns,na)
    Obs = Array{Vector{Int}}(undef,ns,na)
    for si in 1:ns
        for sa in 1:na
            Obs[si,sa] = []
        end
    end

    for (oi,o) in enumerate(O)
        for (si,s) in enumerate(S)
            for (ai,a) in enumerate(A)
                P_obs[oi,si,ai] = Pr_obs(model,o,s,a)
                P_obs[oi,si,ai] > 0 && push!(Obs[si,ai], oi)
            end
        end
    end

    # 2 : Pre-compute all beliefs
    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief([s],[1.0])
        for (ai,a) in enumerate(A)
            for oi in Obs[si,ai]
                o = O[oi]
                b = update(U, b_s, a, o)
                k = findfirst( x -> x==b , B)
                isnothing(k) && (push!(B,b); k=length(B))
                B_idx[si,ai,oi] = k
            end
        end
    end

    # 3 : Compute Q-values bsoa beliefs
    Qs = zeros(Float64, length(B), na)
    
    # π_QMDP = solve(QMDPSolver_alt(), model)
    # for (b_idx, b) in enumerate(B)
    #     for (ai, a) in enumerate(A)
    #         for (si, s) in enumerate(S)
    #             if pdf(b,s) > 0
    #                 Qs[b_idx, ai] += pdf(b,s) * π_QMDP.Q_MDP[si,ai]
    #             end
    #         end
    #     end
    # end
    # println(Qs)

    Qs_prev = deepcopy(Qs)
    for i=1:solver.max_iterations
        for (b_idx,b) in enumerate(B)
            for (ai, a) in enumerate(A)
                Qs[b_idx,ai] = _get_Q_BIB(model,b,a, Qs_prev, B_idx, P_obs, Obs ; ai=ai)
            end
        end
        max_dif = maximum(map(abs, (Qs .- Qs_prev)) ./ (Qs_prev.+1e-10))
        Qs_prev = deepcopy(Qs)
        Qs = zeros(Float64, length(B), na)
        printdb(i, max_dif)
        max_dif < solver.precision && (printdb("breaking after $i iterations:"); break)
    end
    return BIBPolicy(model, BIB_Data(Qs,B_idx,P_obs,Obs) )
end



function _get_Q_BIB(model,b,a,Qs,B_idx,P_obs, Obs; ai=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), actions(model)) )
    S = states(model)
    S_dict = Dict( zip(S, 1:length(S)) )
    Q = reward(model,b,a)
    for (oi, o) in enumerate(observations(model))
        Qo = -Inf
        for (api, ap) in enumerate(actions(model))
            Qoa = 0
            for s in support(b)
                si = S_dict[s]
                if o in Obs[si,ai]
                    p_obs = P_obs[oi,si,ai]
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
_get_Q_BIB(model,b,a,D; ai=nothing) = _get_Q_BIB(model,b,a,D.Q, D.B_idx, D.P_obs, D.Obs; ai)

#########################################
#               Policy:
#########################################

struct BIBPolicy <:Policy
    model::POMDP
    Data::BIB_Data
end
POMDPs.updater(π::BIBPolicy) = DiscreteHashedBeliefUpdater(π.model)

function POMDPs.action(π::BIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = _get_Q_BIB(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return bestA
end