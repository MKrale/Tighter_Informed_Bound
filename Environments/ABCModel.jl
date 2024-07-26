module ABCModel

using POMDPs, POMDPTools, QuickPOMDPs

export ABC

struct ABC <: POMDP{Int,Int,Int} end

function T_int(s,a)
    s==1 && (return SparseCat([2,3], [0.5,0.5]))
    s==4 && (return SparseCat([4], [1]))
    (a==2 || a==3) && (return SparseCat([4], [1]))
    s==2 ? sn = 3 : sn = 2
    return SparseCat([s,sn], [1.0, 0.2])
end

R_int(s::Int,a) = (s!=1 && s==a) ? 1 : 0
O_int(a,sp) = SparseCat([1], [1])

"""A toy model for testing heuristics."""
ABC_ints() = QuickPOMDP(
    states=1:4,     # s_0, s_a, s_b, terminal
    actions=1:3,    # c, a, b
    observations=[1], # nullObs
    discount=0.90,

    transition = T_int,
    observation = O_int,
    reward= R_int,
    initialstate= SparseCat([1], [1]),
    isterminal = s -> s==4
)


function T(s,a)
    s=="init" && (return SparseCat(["A", "B"], [0.5, 0.5]))
    s=="Ap" && (return SparseCat(["A"], [1]))
    s=="Bp" && (return SparseCat(["B"], [1]))
    s=="terminal" && (return SparseCat(["terminal"], [1]))
    (a=="a" || a=="b") && (return SparseCat(["terminal"], [1]))
    s=="A" ? not_s = "B" : not_s = "A"
    return SparseCat([s,not_s], [0.8, 0.2])
end

R(s,a) = ( (s=="A" && a=="a") || (s=="B" && a=="b")) ? 1 : 0
O(a,sp) = SparseCat(["nothing"],[1])

ABC() = QuickPOMDP(
    states = ["init","A","B","Ap","Bp","terminal"],
    actions=["a","b","c"],
    observations=["nothing"],
    discount=0.90,

    transition = T,
    observation = O,
    reward = R,
    initialstate = SparseCat(["init"], [1]),
    isterminal = s -> s=="terminal"
)

end