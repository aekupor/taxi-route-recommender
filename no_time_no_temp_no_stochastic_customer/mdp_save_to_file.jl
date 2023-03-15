using CSV
using DataFrames
using StatsBase
using TabularTDLearning
using POMDPModels
using POMDPTools
using Random
using Plots
using LinearAlgebra
using DelimitedFiles



include("taxi_world_final.jl") 


function read_in_file(inputfilename)
    df = CSV.read(inputfilename, DataFrame)
    return df
end

function convert_state_to_number(s::taxi_world_state) 
    linear = LinearIndices((1:size_x, 1:size_y, 1:2)) #1:4, 1:4, 1:2))
    return linear[s.x, s.y, Int(s.received_request) + 1] #s.temp, s.time, Int(s.received_request) + 1]
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

number_states = size_x * size_y * 2 #4 * 4 
number_actions = 5 

function solve_QLearning(df, model)
    Q_old = copy(model.Q)
    for k in 1:3000
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
                break
            end
        end
        Q_old = copy(model.Q)
    end

    actions = Dict()
    for x in 1:size_x
        for y in 1:size_y
          #  for request in 1:2
            s= convert_state_to_number(taxi_world_state(x, y, 1, 1, false))
            best_value = model.Q[s, 1]
            best_action = 1
            for a in model.A
                if model.Q[s,a] > best_value
                    best_value = model.Q[s,a]
                    best_action = a
                end
            end
            actions[s] = best_action
        end
    end
#end
    writedlm("q_learning_small_notimetemp_alpha_01.txt", actions)
    return actions
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
            break
        end
        Q_old = copy(model.Q)
    end

    
    actions = Dict()
    for x in 1:size_x
        for y in 1:size_y
          #  for request in 1:2
            s= convert_state_to_number(taxi_world_state(x, y, 1, 1, false))
            best_value = model.Q[s, 1]
            best_action = 1
            for a in model.A
                if model.Q[s,a] > best_value
                    best_value = model.Q[s,a]
                    best_action = a
                end
            end
            actions[s] = best_action
        end
    end
#end
    writedlm("sarsa_small_notimetemp_alpha_01.txt", actions)
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

function update_decay!(model::QLearning, s, a, r, sp, epoch)
    Y, Q = model.Y, model.Q
    alpha = model.alpha / (1 + 0.7 * epoch)
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

# function to make heatmap 
function make_heatmap_full(policy, temp, time, heatmap_name)
    heatmap_matrix = zeros(size_x, size_y)
    for x in 1:size_x
        for y in 1:size_y
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
end

function get_action(act_index)
    if act_index == 1
        return :up
    elseif act_index == 2
        return :down
    elseif act_index == 3
        return :left
    elseif act_index == 4
        return :right
    elseif act_index == 5
        return :stay
    end
end;

discount_rate = 0.9
mdp = taxi_world()
train_df = read_in_file("train_dataset_new_small_notimetemp.txt")

# q learning
model = QLearning(collect(1: number_states), collect(1: number_actions), discount_rate, zeros(number_states, number_actions), .01)
#@time solve_QLearning(train_df, model)
q_dict = solve_QLearning(train_df, model)

sarsa_dict = solve_Sarsa(train_df)

#policy_arr = Dict(policy)
#print(policy_arr)
q_policy = FunctionPolicy(s->get_action(q_dict[convert_state_to_number(s)]))

sarsa_policy = FunctionPolicy(s->get_action(sarsa_dict[convert_state_to_number(s)]))
mdp = taxi_world()
short = RolloutSimulator(max_steps=1000)

random_policy = RandomPolicy(mdp)

total_r_1 = 0.0
total_r_2 = 0.0
total_r_3 = 0.0
trials = [1:1000]
q_rewards = []
sarsa_rewards = []
rand_rewards = []

for trial in 1:1000
    mdp_3 = taxi_world()
    mdp_4 = taxi_world()
    mdp_5 = taxi_world()
    rand_x =rand(1:size_x)
    rand_y = rand(1:size_y)
    temp = 1#rand(1:4)
    time = 1#rand(1:4)
    POMDPs.initialstate(mdp_3::taxi_world) = Deterministic(taxi_world_state(rand_x, rand_y, temp, time, false))
    POMDPs.initialstate(mdp_4::taxi_world) = Deterministic(taxi_world_state(rand_x, rand_y, temp, time, false))
    POMDPs.initialstate(mdp_5::taxi_world) = Deterministic(taxi_world_state(rand_x, rand_y, temp, time, false))
    global total_r_1 += simulate(short, mdp_3, q_policy)
    global total_r_2 += simulate(short, mdp_4, random_policy)
    global total_r_3 += simulate(short, mdp_4, sarsa_policy)
    push!(q_rewards, total_r_1)
    push!(rand_rewards, total_r_2)
    push!(sarsa_rewards, total_r_3)

end

println("The total reward for Q-learning is $(total_r_1 / 1000)\n")
println("The total reward for random is $(total_r_2 / 1000)\n")
println("The total reward for sarsa is $(total_r_3 / 1000)\n")

plot(trials, [q_rewards rand_rewards sarsa_rewards], label=["Total Rewards from Q-learning" "Total Rewards from a Random Policy" "Total Rewards from SARSA"], linewidth=3)
savefig("myplot_NEW.png")   


#policy = CSV.read("q_learning_small_alpha_01.txt", DataFrame)
# make heatmaps
 #for time in 1:4
 #    for temp in 1:4 
         #make_heatmap_full(policy, 1, 1, "heatmap_$(time + temp)")
  #   end
# end



