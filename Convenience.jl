function POMDPs.reward(model::POMDP, b::DiscreteHashedBelief,a)
    r = 0.0
    for (s,p) in zip(b.state_list, b.probs)
        s == POMDPTools.ModelTools.TerminalState() || ( r += p * POMDPs.reward(model,s,a) )
    end
    return r
end
# POMDPs.reward(model::POMDP, b::DiscreteHashedBelief,a) = sum( (s,p) ->  p * POMDPs.reward(model,s,a), zip(b.state_list, b.probs); init=0.0)

function Pr_obs(model::POMDP, o,s,a)
    p = 0
    for (sp, psp) in weighted_iterator(transition(model,s,a))
        dist = observation(model,a,sp)
        p += psp*pdf(dist,o)
    end
    return p
end

Pr_obs(model::POMDP, o, b::DiscreteHashedBelief, a) = sum( (s,p) -> p*Pr_obs(model,o,s,a), zip(b.state_list, b.probs) )

function add_to_dict!(dict, key, value; func=+, minvalue=0)
    if haskey(dict, key)
        dict[key] = func(dict[key], value)
    elseif isnothing(minvalue) || value > minvalue
        dict[key] = value
    end
end

