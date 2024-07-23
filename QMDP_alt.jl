struct QMDPSolver_alt <: Solver
    precision::AbstractFloat
    max_iterations::Int
end

QMDPSolver_alt() = QMDPSolver_alt(0, 50)

struct QMDPPlanner_alt <: Policy
    Model::POMDP
    Q_MDP::Matrix{AbstractFloat}
    V_MDP::Vector{AbstractFloat}
end

"""Computes the QMDP table using value iteration"""
function solve(sol::QMDPSolver_alt, m::POMDP)
    Q = zeros((length(states(m)),length(actions(m))))
    Qmax = zeros(length(states(m)))
    i=0
    # Lets iterate!
    largest_change = Inf
    i=0
    while (largest_change > sol.precision) && (i < sol.max_iterations)
        i+=1
        largest_change = 0
        S = reverse(sortperm(Qmax)) #TODO: this can maybe be made even more efficient by also keeping track of how far states are from the initial state & updating in reverse order to that?
        for (si,s) in enumerate(states(m))
            for (ai,a) in enumerate(actions(m))
                Qnext = reward(m,s,a)
                thisT = transition(m,s,a)
                for (spi, sp) in enumerate(states(m))
                    Qnext += pdf(thisT, sp) * discount(m) * Qmax[spi]
                end
                # largest_change = max(largest_change, abs(Q[si,ai] / Qnext), abs(2- Q[si,ai] / Qnext))
                largest_change = max(largest_change, abs(Q[si,ai]-Qnext) / (Qnext+1e-10) )
                Q[si,ai] = Qnext
                Qmax[si] = max(Qmax[si], Q[si,ai])
            end
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