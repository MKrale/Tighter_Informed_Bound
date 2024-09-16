module ABCModel

using POMDPs, POMDPTools, QuickPOMDPs

export ABC

# function

function T(s,a)
    s=="init" && (return SparseCat(["A", "B"], [0.5, 0.5]))
    # s=="Ap" && (return SparseCat(["A"], [1]))
    # s=="Bp" && (return SparseCat(["B"], [1]))
    s=="terminal" && (return SparseCat(["terminal"], [1]))
    (a=="a" || a=="b") && (return SparseCat(["terminal"], [1]))
    s=="A" ? not_s = "B" : not_s = "A"
    return SparseCat([s,not_s], [0.8, 0.2])
end

R(s,a) = ( (s=="A" && a=="a") || (s=="B" && a=="b")) ? 1 : 0
O(a,sp) = SparseCat(["nothing"],[1])

ABC(;discount=0.95) = QuickPOMDP(
    states = ["init","A","B","Ap","Bp","terminal"],
    # states = ["init","A","B","terminal"],
    actions=["a","b","c"],
    observations=["nothing"],
    discount=discount,

    transition = T,
    observation = O,
    reward = R,
    initialstate = SparseCat(["init"], [1]),
    isterminal = s -> s=="terminal"
)

end