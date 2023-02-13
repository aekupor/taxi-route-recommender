using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat


struct taxiMDP
    Y # discount factor
    S # state space
    A # action space
    T # transition function
    R # reward function
    #TR # sample transition and reward
end

function lookahead(mdp::taxiMDP, U, s, a)
    S, T, R, Y = mdp.S, mdp.T, mdp.R, mdp.Y
    return R(s,a) + Y*sum(T(s,a,s_new)*U(s_new) for s_new in S)
end

reward = function (s, a)  #TODO: update
    if a == "north"
        return -1.0
    elseif s == "south"
        return -100.0
    else 
        return 10.0
    end
end

utility = function(s)  # TODO: update
    return 1
end

trans = function (s, a, s_new)  #this data is from running the simulator and printing get_transition_probs
    if s == "golden_gate_park"
        if s_new == "golden_gate_park"
            return 0.2813690465079768
        elseif s_new == "sunset"
            return 0.06397271130176105
        elseif s_new == "bayview"
            return 0.2834367895672558
        elseif s_new == "mission"
            return 0.3081994002176982
        elseif s_new == "fishermans_wharf"
            return 0.06302205240530818
        end
    elseif s == "sunset"
        if s_new == "golden_gate_park"
            return 0.17056643739636274
        elseif s_new == "sunset"
            return 0.2492636296055344
        elseif s_new == "bayview"
            return 0.13981410701151556
        elseif s_new == "mission"
            return 0.2555662039617832
        elseif s_new == "fishermans_wharf"
            return 0.18478962202480415
        end
    elseif s == "bayview"
        if s_new == "golden_gate_park"
            return 0.22039135897652012
        elseif s_new == "sunset"
            return 0.024652871735974927
        elseif s_new == "bayview"
            return 0.35630318683137047
        elseif s_new == "mission"
            return 0.3423632027601907
        elseif s_new == "fishermans_wharf"
            return 0.056289379695943775
        end
    elseif s == "mission"
        if s_new == "golden_gate_park"
            return 0.08582860046524174
        elseif s_new == "sunset"
            return 0.38370020309117253
        elseif s_new == "bayview"
            return 0.07821975838298886
        elseif s_new == "mission"
            return 0.1431001320578924
        elseif s_new == "fishermans_wharf"
            return 0.3091513060027044
        else
            return 0
        end
    elseif s == "fishermans_wharf"
        if s_new == "golden_gate_park"
            return 0.39041617681739876
        elseif s_new == "sunset"
            return 0.29611552123293966
        elseif s_new == "bayview"
            return 0.059639571425333014
        elseif s_new == "mission"
            return 0.10397806641026408
        elseif s_new == "fishermans_wharf"
            return 0.1498506641140645
        else
            return 0
        end
    end
    return 0
end


myTaxiMDP = taxiMDP(0.95, ["golden_gate_park", "sunset", "bayview", "mission", "fishermans_wharf"], ["north", "south", "east", "west", "stay"], trans, reward)

result = lookahead(myTaxiMDP, utility, "mission", "north")
print(result)
