# tolerance is ||alpha^k - alpha^k+1||_infty


#########################################
#               Solver:
#########################################

mutable struct BIBSolver <: Solver
    max_iterations::Int64       # maximum iterations taken by solver
    precision::Float64          # precision at which iterations is stopped
end

function BIBSolver(;max_iterations::Int64=100, precision::Float64=1e-3)
    return BIBSolver(max_iterations, precision)
end

function POMDPs.solve(solver::BIBSolver, model::POMDP)

    S, A, O = states(model), actions(model), observations(model) #TODO: figure out if I need to use ordered_... here
    println(S, " ", A)
    ns, na, no = length(states(model)), length(actions(model)), length(observations(model))
    # Bs = Array{DiscreteBelief,3}(undef, ns,na,no)
    
    B_idx = zeros(Int,ns,na,no)
    B = Array{DiscreteHashedBelief,1}()
    U = DiscreteHashedBeliefUpdater(model)

    # 1: Precompute observatin probabilities
    P_obs = zeros(no,ns,na)
    for (oi,o) in enumerate(O)
        for (si,s) in enumerate(S)
            for (ai,a) in enumerate(A)
                P_obs[oi,si,ai] = Pr_obs(model,o,s,a)
            end
        end
    end

    # 2 : Pre-compute all beliefs
    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief([s],[1.0])
        for (ai,a) in enumerate(A)
            for (oi,o) in enumerate(O)
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
                Qs[b_idx,ai] = _get_Q_BIB(model,b,a,Qs_prev,B_idx;P_obs=P_obs, ai=ai)
            end
        end
        max_dif = maximum(map(abs, (Qs .- Qs_prev)) ./ (Qs_prev.+1e-10))
        # max_dif = maximum( (x,y) -> (iszero(y) ? 0 : abs(x-y)/y), zip(Qs, Qs_prev))
        Qs_prev = deepcopy(Qs)
        Qs = zeros(Float64, length(B), na)
        printdb(i, max_dif)
        max_dif < solver.precision && (printdb("breaking after $i iterations:"); break)
    end
    return BIBPolicy(model, Qs_prev, B_idx)
end

function _get_Q_BIB(model,b,a,Qs,B_idx;P_obs=nothing, ai=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), actions(model)) )
    Q = reward(model,b,a)
    for (oi, o) in enumerate(observations(model))
        Qo = -Inf
        for (api, ap) in enumerate(actions(model))
            Qoa = 0
            for (si, s) in enumerate(states(model))
                if pdf(b,s) > 0
                    isnothing(P_obs) ? ( p_obs = Pr_obs(model,o,s,a) ) : ( p_obs = P_obs[oi,si,ai] )
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


#########################################
#               Policy:
#########################################

struct BIBPolicy <:Policy
    model::POMDP
    Q::Array{Float64,2}
    B::Array{Int,3}
end
POMDPs.updater(π::BIBPolicy) = DiscreteHashedBeliefUpdater(π.model)

function POMDPs.action(π::BIBPolicy, b)
    model, Q, B = π.model, π.Q, π.B
    b = DiscreteHashedBelief(b)
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(actions(model))
        Qa = _get_Q_BIB(model, b, a, Q, B; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return bestA
end


#########################################
#               Old Code:
#########################################

# # 2: For all beliefs, pre-compute next beliefs
# nb_base = length(B)
# Bnext_idx = Array{Tuple{Float64,Int}}(undef, nb_base, na, no)

# for b_idx=1:nb_base
#     b = B[b_idx]
#     for (api,ap) in enumerate(A)
#         for (opi, op) in enumerate(O)
#             bp = update(U,b,ap,op)
#             pbp = Pr_obs(model,o,b,ap)
#             k = findfirst( x -> x==bp , B)
#             isnothing(k) && (append!(B,bp); k=length(B))
#             Bnext_idx[b_idx,api, opi] = (pbp, k)
#         end
#     end
# end

        #     Q = reward(model,b,a)
        #     for (oi, o) in enumerate(O)
        #         Qo = -Inf
        #         for (api, ap) in enumerate(A)
        #             Qoa = 0
        #             for (si, s) in enumerate(S)
        #                 if pdf(b,s) > 0
        #                     p = pdf(b,s) * P_obs[oi,si,ai]
        #                     bp_idx = B_idx[si,ai,oi]
        #                     Qoa += p * Qs_prev[bp_idx,api]
        #                 end
        #             end
        #             Qo = maximum(Qo, Qoa)
        #         end
        #         Q += discount(model) * Qo
        #     end
        #     Qs[b_idx, ai] = Q