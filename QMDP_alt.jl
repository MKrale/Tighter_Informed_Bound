struct QMDPSolver_alt <: Solver
    precision::AbstractFloat
    max_iterations::Int
end

QMDPSolver_alt() = QMDPSolver_alt(1e-10, 5000)

struct QMDPPlanner_alt <: Policy
    Model::POMDP
    Q_MDP::Matrix{AbstractFloat}
    V_MDP::Vector{AbstractFloat}
end

function get_max_r(m::POMDP)
    maxr = 0
    for s in states(m)
        for a in actions(m)
            maxr = max(maxr, reward(m,s,a))
        end
    end
    return maxr 
end


"""Computes the QMDP table using value iteration"""
function solve(sol::QMDPSolver_alt, m::POMDP)
    Q = zeros((length(states(m)),length(actions(m))))
    Qmax = zeros(length(states(m)))
    max_r = get_max_r(m)
    maxQ = max_r / (1-discount(m))
    Q[:,:] .= maxQ
    Qmax[:] .= maxQ

    i=0
    # Lets iterate!
    largest_change = Inf
    i=0
    S_dict = Dict( zip(states(m), 1:length(states(m))))
    while (largest_change > sol.precision) && (i < sol.max_iterations)
        i+=1
        largest_change = 0
        for (si,s) in enumerate(states(m))
            for (ai,a) in enumerate(actions(m))
                Qnext = reward(m,s,a)
                thisT = transition(m,s,a)
                for sp in support(thisT)
                    Qnext += pdf(thisT, sp) * discount(m) * Qmax[S_dict[sp]]
                end
                largest_change = max(largest_change, abs((Qnext - Q[si,ai]) / (Q[si,ai]+1e-10) ))
                Q[si,ai] = Qnext
            end
            Qmax[si] = maximum(Q[si,:])
        end
    end
    
    return QMDPPlanner_alt(m,Q,Qmax)
end

"""Picks action with highest value in QMDP table"""
function action(π::QMDPPlanner_alt,b; returnValue=false)
    M = π.Model
    thisQ = zeros(length(actions(M)))
    for a in actions(M)
        for s in support(b)
            thisQ[a] += pdf(b,s) * π.Q_MDP[s,a]
        end
    end 
    amax = argmax(thisQ)
    if returnValue
        return (amax, thisQ[amax])
    else
        return amax
    end
end