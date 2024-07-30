using JuMP, SCS, GLPK, Gurobi, Convex
column(model, x) = Cint(Gurobi.column(backend(model), index(x)) - 1)

# Qs = [2,2,2,4,4]
# b = [0.1, 0.4, 0.5, 0.0]
# Bs=[[1. 0. 0. 0. ]
#     [0. 1. 0. 0. ]
#     [0. 0. 1. 0. ]
#     [0. 0. 0. 1. ]
#     [0. 0.5 0.5 0.]]

# model = direct_model(Gurobi.Optimizer())
# @variable(model, 0.0 <= x[1:5] <= 1)
# @variable(model, t[1:5])

# for s=1:4
#     Idx, Ps = [], []
#     for (bpi, bp) in enumerate(eachrow(Bs))
#         if bp[s] > 0
#             push!(Idx, bpi)
#             push!(Ps, bp[s])
#         end
#     end
#     @constraint(model, sum(x[Idx[i]] * Ps[i] for i in 1:length(Idx)) == b[s])
# end

# for i=1:5
#     GRBaddgenconstrLog(backend(model), "logCost", column(model, x[i]), column(model, t[i]), "")
# end
# @objective(model, Max, sum(-t.*x))
# print(model)
# optimize!(model)
# @assert is_solved_and_feasible(model)
# solution_summary(model)
# println(map(value, x))


# Qs = [2,2,2,4,4]
# b = [0.1, 0.4, 0.5, 0.0]
# Bs=[[1. 0. 0. 0. ]
#     [0. 1. 0. 0. ]
#     [0. 0. 1. 0. ]
#     [0. 0. 0. 1. ]
#     [0. 0.5 0.5 0.]]

# x = Variable(5)
# add_constraint!(x, sum(x) == 1)
# add_constraint!(x, x>0)

# norm_constraints = []
# for s=1:4
#     Idx, Ps = [], []
#     for (bpi, bp) in enumerate(eachrow(Bs))
#         if bp[s] > 0
#             push!(Idx, bpi)
#             push!(Ps, bp[s])
#         end
#     end
#     push!(norm_constraints, sum(x[Idx[i]] * Ps[i] for i in 1:length(Idx)) == b[s])
# end

# problem = maximize(sum(entropy(x)))
# solve!(problem, Gurobi.Optimizer)




#################################




# S_dict = Dict( zip(states(model), 1:length(states(model))) )
# Qs_Bbao = zeros(length(Bbao_data.Bbao))
# for (bbaoi, bbao) in enumerate(Bbao_data.Bbao)
#     Qs = zeros(constants.na)
#     bi, ai, oi = Bbao_data.Bbao_representatives
#     for (api, ap) in enumerate(constants.A)
#         thisQa = 0
#         for (s, ps) in weighted_iterator(b)
#             p = pdf(b,s)
#             thisQa += p * Qs[bp_idx,api]
#         end
#         thisQ = max(thisQ, thisQa)
#     end
#     Qs_Bbao[bbaoi] = thisQ
# end

# Qs_new = zero(Qs) # TODO: this may be inefficient?
# for (bi, b) in enumerate(B)
#     for (ai, a) in enumerate(constants.A)
#         Qba = breward(model,b,a)
#         for (oi, o) in enumerate(constants.O)
#             Qbao = 0
#             (in_B, boai) = Bbao_data.Bbao_idx[bi, ai, oi]
#             if in_B
#                 Qbao = maximum(Qs[boai, :])
#             else
#                 Qbao = Qs_Bbao[boai]
#             end
#             po = sum(s -> pdf(b,s) * SAO_probs[oi,S_dict[s],ai], support(b))
#             Qba += po * discount(model) * Qbao
#         end
#         Qs_new[bi,ai] = Qba
#     end
# end