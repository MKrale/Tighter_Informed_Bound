module GuessingChains

using POMDPs, POMDPTools, QuickPOMDPs

export GuessingChain

# function

N=10

ps_initial = []
for s in 1:10
    push!(ps_initial, 0.5^s)
end

function T(s,a)
    # s=="init" && (return SparseCat(["A", "B"], [0.5, 0.5]))
    # s=="Ap" && (return SparseCat(["A"], [1]))
    # s=="Bp" && (return SparseCat(["B"], [1]))
    # a=="w" && return SparseCat([s], [1])
    s==N && return SparseCat([0,s],[0.5,0.5])
    return SparseCat([0,s+1], [0.5,0.5])
end

function R(s,a)
    (s==0 && a=="x") && return -100
    (s>0 && a=="y") && return 1
    return 0
end
function O(a,sp) = SparseCat(["nothing"],[1.0])

GuessingChain(;discount=0.95) = QuickPOMDP(
    states = 0:N,
    # states = ["init","A","B","terminal"],
    actions=["x","y","w"],
    observations=["nothing"],
    discount=discount,

    transition = T,
    observation = O,
    reward = R,
    initialstate = SparseCat(1:N, ps_initial),
)

end