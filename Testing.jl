# @kwdef struct STIBSolver <: TIBSolver
#     max_iterations::Int64   = 250       # maximum iterations taken by solver
#     max_time::Float64       = 3600      # maximum time spent solving
#     precision::Float64      = 1e-4      # precision at which iterations is stopped
#     precomp_solver          = FIBSolver_alt(precision=1e-4, max_iterations=250, max_time=360)
#     STIBSolver(;max_iterations=max_iterations, max_time=max_time, precision=precision) = new(max_iterations, max_time, precision, FIBSolver_alt(precision=precision, max_time=max_time, max_iterations=max_iterations))
# end

# @kwdef struct testing
#     field1::Float64 = 4
#     field2::Float64 = 8
# end



# println(testing().field2)
# testing(field1) = testing(field1, field1/2)
# println(testing().field2)
# println(testing(field1=5).field2)
# println(testing(field1=5, field2=5).field2)


# @kwdef struct SomeParams
#     A::Float64 = 3
#     B::Float64 = 4
#     SomeParams(C::Float64) = new(C^2, C + 0.5)
#     SomeParams(A::Float64, B::Float64) = new(A, B)
# end

# println(SomeParams(3.0))
# println(SomeParams(;A=3.0, B=2.0))



function get_minratio_weights(b, B; Bdao_data=nothing)
    



function get_closeness_weights(b, B; Bbao_data=nothing)
    if !isnothing(Bbao_data)
        closest_belief = get_closest_belief(b,Bbao_data.Bbao_overlap)
        for s in support(b)

    else
        println("Error: not implemented!")
    end




