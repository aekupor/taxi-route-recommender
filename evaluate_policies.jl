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

number_states = size_x * size_y * 4 * 4 * 2

function convert_state_to_number(s::taxi_world_state) 
    linear = LinearIndices((1:size_x, 1:size_y, 1:4, 1:4, 1:2))
    return linear[s.x, s.y, s.temp, s.time, Int(s.received_request) + 1]
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

mdp = taxi_world()
# state_space = states(mdp);
# POMDPs.actions(mdp::taxi_world) = [:up, :down, :left, :right, :stay];
actions = ones(number_states)
one_policy = FunctionPolicy(s->get_action(actions[convert_state_to_number(s)]))
POMDPs.initialstate(mdp::taxi_world) = Deterministic(taxi_world_state(3, 4, 1, 1, false))
# for (s,a,r, sp) in stepthrough(mdp, one_policy, "s,a,r,sp", max_steps=5)
#     @show s
#     @show a
#     @show r
#     @show sp
#     println()
# end

# read CSV file into a DataFrame
policy = CSV.read("Qlearning3000.policy", DataFrame)
policy_arr = Matrix(policy)
qpolicy = FunctionPolicy(s->get_action(policy_arr[convert_state_to_number(s)]))
mdp_2 = taxi_world()
short = RolloutSimulator(max_steps=1000)

random_policy = RandomPolicy(mdp_2)

total_r_1 = 0.0
total_r_2 = 0.0
trials = [1:1000]
q_rewards = []
rand_rewards = []
for trial in 1:1000
    mdp_3 = taxi_world()
    mdp_4 = taxi_world()
    rand_x = rand(1:100)
    rand_y = rand(1:100)
    temp = rand(1:4)
    time = rand(1:4)
    POMDPs.initialstate(mdp_3::taxi_world) = Deterministic(taxi_world_state(rand_x, rand_y, temp, time, false))
    POMDPs.initialstate(mdp_4::taxi_world) = Deterministic(taxi_world_state(rand_x, rand_y, temp, time, false))
    global total_r_1 += simulate(short, mdp_3, qpolicy)
    global total_r_2 += simulate(short, mdp_4, random_policy)
    push!(q_rewards, total_r_1)
    push!(rand_rewards, total_r_2)
end

print("The total reward for Q-learning is $(total_r_1 / 1000)\n")
# total_r_2 = 0.0
# for trial in 1:1000
#     mdp_4 = taxi_world()
#     POMDPs.initialstate(mdp_4::taxi_world) = Deterministic(taxi_world_state(1, 1, 1, 1, false))
#     global total_r_2 += simulate(short, mdp_4, random_policy)
# end

print("The total reward for random is $(total_r_2 / 1000)\n")
plot(trials, [q_rewards rand_rewards], label=["Total Rewards from Q-learning" "Total Rewards from a Random Policy"], linewidth=3)
savefig("myplot_2.png")   
# q_r = 0.0
# POMDPs.initialstate(mdp::taxi_world) = Deterministic(taxi_world_state(1, 1, 1, 1, false))
# for (s,a,r, sp) in stepthrough(mdp, qpolicy, "s,a,r,sp", max_steps=1000)
#     global q_r += r
# end
# println()
# print(q_r)
# println()

# rand_r = 0.0
# POMDPs.initialstate(mdp::taxi_world) = Deterministic(taxi_world_state(1, 1, 1, 1, false))
# for (s,a,r, sp) in stepthrough(mdp, random_policy, "s,a,r,sp", max_steps=1000)
#     global rand_r += r
# end

# print(rand_r)
# println()