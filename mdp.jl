using CSV
using DataFrames
using StatsBase


include("taxi_world_test") 


function read_in_file(inputfilename)
    df = CSV.read(inputfilename, DataFrame)
    return df
end

#ashlee todo: change to LinearIndices/CartesianIndices. see doc on project 1
function covert_state_to_number(s::taxi_world_state) 
    # special cases to handle not going over 100 states (since 1010 > 100)
    if s.x == 10 && s.y == 10 
        return 1
    elseif s.x == 10
        return s.y
    elseif s.y == 10
       return s.x
    end
    number = string(s.x) * string(s.y) # todo: add in time, temp, request later 
    return parse(Int64, number)
end

function convert_number_to_action(a::Int64) 
    # return the symbol of a from the number
    if a==1
        return :up
    elseif a==2
        return :down
    elseif a==3
        return :left
    elseif a==4
        return :right
    elseif a==5
        return :stay
    end
end

function solve_QLearning(df)
    # number of states = 10x * 10y (add in temp, time, request later)
    number_states = 100
    number_actions = 5
    
    model = QLearning(collect(1: number_states), collect(1: number_actions), discount_rate, zeros(number_states, number_actions), .01)
    for k in 1:100
        for dfr in eachrow(df)
            s = taxi_world_state(dfr.x, dfr.y, dfr.temp, dfr.time, dfr.received_request)
            sp = taxi_world_state(dfr.sp_x, dfr.sp_y, dfr.sp_temp, dfr.sp_time, dfr.sp_received_request)
            model = update!(model, covert_state_to_number(s), dfr.a, dfr.r, covert_state_to_number(sp))
        end
    end

    actions = Vector{Int}()
    for s in model.S
        best_value = model.Q[s, 1]
        best_action = 1
        for a in model.A
            if model.Q[s,a] > best_value
                best_value = model.Q[s,a]
                best_action = a
            end
        end
        push!(actions, best_action)
    end
    return actions
end

mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    Y # discount
    Q # action value function
    alpha # learning rate
end

function lookahead(model::QLearning, s, a)
    return model.Q[s,a]
end

function update!(model::QLearning, s, a, r, sp)
    Y, Q, alpha = model.Y, model.Q, model.alpha
    Q[s,a] += alpha*(r + Y*maximum(Q[sp,:]) - Q[s,a])
    return model
end

function evaluate_policy(df, policy, mdp)
    total_u = 0
    current_row = 0
    for dfr in eachrow(df)
        s = taxi_world_state(dfr.x, dfr.y, dfr.temp, dfr.time, dfr.received_request)
        s_num = covert_state_to_number(s)
        a = convert_number_to_action(policy[s_num])
        transitions = POMDPs.transition(mdp, s, a)

        neighbors = Vector{taxi_world_state}()
        weights = Vector()
        for item in transitions
            push!(neighbors, item[1])
            push!(weights, item[2])
        end

        # a bit janky, but it works
        num = rand()
        prob_total = 0
        index = 0
        for weight in weights
            if num > prob_total
                index += 1
            end
            prob_total += weight
        end

        sp = neighbors[index] 
        total_u += (discount_rate ^ current_row) * POMDPs.reward(mdp, s, a, sp) 
        current_row += 1
        if sp.received_request == true 
            current_row = 0
        end
    end
    return total_u
end

function generate_random_policy()
    actions = []
    for i in 1:100
        num = rand()
        if num < .2
            push!(actions, 1)
        elseif num < .4
            push!(actions, 2)
        elseif num < .6
            push!(actions, 3)
        elseif num < .8
            push!(actions, 4)
        else
            push!(actions, 5)
        end
    end
    return actions
end

discount_rate = 0.95
mdp = taxi_world()

df = read_in_file("dataset.txt")
test_data = read_in_file("test_dataset.txt")

qlearning_policy = solve_QLearning(df)
total_u_qlearning = evaluate_policy(test_data, qlearning_policy, mdp)
println(total_u_qlearning)

random_policy = generate_random_policy()
total_u_random_policy = evaluate_policy(test_data, random_policy, mdp)
println(total_u_random_policy)

