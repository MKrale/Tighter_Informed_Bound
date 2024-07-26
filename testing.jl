using JuMP, HiGHS

Qs = [2,2,2,4,4]
b = [0.1, 0.4, 0.5, 0.0]
Bs=[[1. 0. 0. 0. ]
    [0. 1. 0. 0. ]
    [0. 0. 1. 0. ]
    [0. 0. 0. 1. ]
    [0. 0.5 0.5 0.]]

model = Model(HiGHS.Optimizer)
@variable(model, x[1:5] in Semicontinuous(0.0,1.0))
for s=1:4
    Idx, Ps = [], []
    for (bpi, bp) in enumerate(eachrow(Bs))
        if bp[s] > 0
            push!(Idx, bpi)
            push!(Ps, bp[s])
        end
    end
    @constraint(model, sum(x[Idx[i]] * Ps[i] for i in 1:length(Idx)) == b[s])
end

@objective(model, Max, sum(Qs .* x))
print(model)
optimize!(model)
@assert is_solved_and_feasible(model)
solution_summary(model)
println(map(value, x))