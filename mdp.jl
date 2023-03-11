using CSV
using DataFrames
using StatsBase
using TabularTDLearning
using POMDPModels
using POMDPTools
using Random
using Plots
using LinearAlgebra
using Printf

include("taxi_world_final.jl") 

## function to read in file 
function read_in_file(inputfilename)
    df = CSV.read(inputfilename, DataFrame)
    return df
end


function write_policy(policy, policy_filename)
    open(policy_filename, "w") do io
        for n in eachindex(policy)
            @printf(io, "%s\n", policy[n])
        end
    end
end

function convert_state_to_number(s::taxi_world_state) 
    linear = LinearIndices((1:size_x, 1:size_y, 1:4, 1:4, 1:2))
    return linear[s.x, s.y, s.temp, s.time, Int(s.received_request) + 1]
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

number_states = size_x * size_y * 4 * 4 * 2
number_actions = 5 

function solve_QLearning(df, model, k_max)
    Q_old = copy(model.Q)
    for k in 1:k_max
        for dfr in eachrow(df)
            s = taxi_world_state(dfr.x, dfr.y, dfr.temp, dfr.time, dfr.received_request)
            sp = taxi_world_state(dfr.sp_x, dfr.sp_y, dfr.sp_temp, dfr.sp_time, dfr.sp_received_request)
            model = update!(model, convert_state_to_number(s), dfr.a, dfr.r, convert_state_to_number(sp))
        end
        if mod(k, 100) == 0
            diff = norm(Q_old - model.Q)
            println(diff)
            if diff < 1
                print("The epoch number with difference smaller than 1 is $k")
            end
        end
        Q_old = copy(model.Q)
    end
    actions = mapslices(argmax,model.Q,dims=2)
    return model, actions
end

function solve_Sarsa(df)
    model = Sarsa(collect(1: number_states), collect(1: number_actions), discount_rate, zeros(number_states, number_actions), .01, nothing)
    Q_old = copy(model.Q)
    for k in 1:3000
        for dfr in eachrow(df)
            s = taxi_world_state(dfr.x, dfr.y, dfr.temp, dfr.time, dfr.received_request)
            sp = taxi_world_state(dfr.sp_x, dfr.sp_y, dfr.sp_temp, dfr.sp_time, dfr.sp_received_request)
            model = update!(model, convert_state_to_number(s), dfr.a, dfr.r, convert_state_to_number(sp))
        end
        if mod(k, 100) == 0
            println(norm(Q_old - model.Q))
        end
        Q_old = copy(model.Q)
    end
    actions = mapslices(argmax,model.Q,dims=2)
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

mutable struct Sarsa
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    Y # discount
    Q # action value function
    a # learning rate
    l # most recent experience tuple (s,a,r)
end

lookahead(model::Sarsa, s, a) = model.Q[s,a]
    
function update!(model::Sarsa, s, a, r, sp)
    if model.l != nothing
        Y, Q, alpha, l = model.Y, model.Q, model.a, model.l
        model.Q[l.s,l.a] += alpha*(l.r + Y*Q[s,a] - Q[l.s,l.a])
    end
    model.l = (s=s, a=a, r=r)
    return model
end

function evaluate_policy(df, policy, mdp, type)
    total_u = 0
    current_row = 0
    for dfr in eachrow(df)
        s = taxi_world_state(dfr.x, dfr.y, dfr.temp, dfr.time, dfr.received_request)
        s_num = convert_state_to_number(s)
        a = 1
        if type == "value_table"
            a = findmax(policy.value_table[s_num])[2]
        elseif type == "random"
            a = rand(1:5)
        else 
            a = policy[s_num]
        end
        a = convert_number_to_action(a)
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
    actions = Dict()
    for i in 1:number_states
        num = rand()
        if num < .2
            actions[i] = 1 
        elseif num < .4
            actions[i] = 2
        elseif num < .6
            actions[i] = 3
        elseif num < .8
            actions[i] = 4
        else
            actions[i] = 5
        end
    end
    return actions
end

discount_rate = 0.9
mdp = taxi_world()

# function to make heatmap 
function make_heatmap_full(policy, temp, time, heatmap_name)
    heatmap_matrix = zeros(100, 100)
    for x in 1:100
        for y in 1:100
            linear = LinearIndices((1:size_x, 1:size_y, 1:4, 1:4, 1:2))
            state_num = linear[x, y, temp, time, 1]
            action = policy[state_num]
            heatmap_matrix[x, y] = action
        end
    end

    heatmap(1:size(heatmap_matrix,1), 1:size(heatmap_matrix,2), heatmap_matrix, c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="x values", ylabel="y values", title="Heatmap for temp = $temp and time = $time", cgrad = cgrad(:matter, 5, categorical = true)
    )
    savefig(heatmap_name)
    return heatmap_matrix
end


df_full = read_in_file("taxi-route-recommender/full_dataset.txt")
println("Done")
# shuffle the dataset
df_permuted = df_full[shuffle(1:size(df_full, 1)), :]
println("Done")
# Split the dataframe into train and test sets
train_size = 0.8
n_train = Int(round(train_size * size(df_permuted, 1)))
train_df = df_permuted[1:n_train, :]
println("Done")
test_df = df_permuted[n_train+1:end, :]
println("Done")


# q learning
model = QLearning(collect(1: number_states), collect(1: number_actions), discount_rate, zeros(number_states, number_actions), .01)
model, qlearning_policy = solve_QLearning(train_df, model, 3000)
write_policy(qlearning_policy, "Qlearning3000.policy")

heatmap_matrix = zeros(100, 100)

# make heatmaps
for time in 1:4
    for temp in 1:4 
        global heatmap_matrix = make_heatmap_full(qlearning_policy, temp, time, "heatmap_$((time - 1) * 4 + temp)")
    end
end

zoomed = heatmap_matrix[45:54, 45:54]

heatmap(1:size(zoomed,1), 1:size(zoomed,2), zoomed , c=cgrad([:blue, :white,:red, :yellow]),
xlabel="x values", ylabel="y values", title="Heatmap for temp = 1 and time = 0")
savefig("zoomed_in.png")

total_u_qlearning = evaluate_policy(test_df, qlearning_policy, mdp, "normal")
println("my q learning")
println(total_u_qlearning)

# # random
# random_policy = generate_random_policy()
# total_u_random_policy = evaluate_policy(test_df, random_policy, mdp, "normal")
# println("my random")
# println(total_u_random_policy)

# #sarsa
# sarsa_policy = solve_Sarsa(train_df)
# total_u_sarsa = evaluate_policy(test_df, sarsa_policy, mdp, "normal")
# println("my sarsa")
# println(total_u_sarsa)

