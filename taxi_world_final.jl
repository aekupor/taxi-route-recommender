# first import the POMDPs.jl interface
using POMDPs
using Distributions
using Random
using DelimitedFiles
using POMDPModels


#println("Hello")
# POMDPModelTools has tools that help build the MDP definition
using POMDPModelTools
# POMDPPolicies provides functions to help define simple policies
using POMDPPolicies
# POMDPSimulators provide functions for running MDP simulations
using POMDPSimulators

size_x = 100
size_y = 100

struct taxi_world_state
    x::Int64 # x position
    y::Int64 # y position
    temp::Int64
    time::Int64
    received_request::Bool # have we already got a request
end

# Initial State Constructor 
taxi_world_state(x::Int64, y::Int64, temp::Int64, time::Int64) = taxi_world_state(x, y, temp, time, false)

# the taxi world MDP type
mutable struct taxi_world <: MDP{taxi_world_state, Symbol}
    size_x::Int64 # x size of grid 
    size_y::Int64 # y size of grid 
    reward_for_request::Float64
    reward_for_gas::Float64
    discount_factor::Float64 # discount factor 
    # add initial state 
end

# we use key worded arguments so we can change any of the values we pass in 


function taxi_world(;sx::Int64=size_x, # size_x
    sy::Int64=size_y, # size_y
    rfr::Float64=1000., # reward for request
    rfg::Float64=-5., #reward for gas
    discount_factor::Float64=0.9)
return taxi_world(sx, sy, rfr, rfg, discount_factor)
end

function POMDPs.states(mdp::taxi_world)
    s = taxi_world_state[] # initialize an array of GridWorldStates
    # loop over all our states
    temps = [1,2,3,4] # corresponds to 45, 60, 75, 90 degree F
    times = [1,2,3,4] # corresponds to 0, 6, 12, 18 (military time)
    for d = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x, time in times, temp in temps
        push!(s, taxi_world_state(x, y, temp, time, d))
    end
    return s
end;

POMDPs.actions(mdp::taxi_world) = [:up, :down, :left, :right, :stay];

# transition helpers
function inbounds(mdp::taxi_world,x::Int64,y::Int64)
    if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
        return true
    else
        return false
    end
end

inbounds(mdp::taxi_world, state::taxi_world_state) = inbounds(mdp, state.x, state.y);

# transition helper which determines distance 
function distance(x1::Int64, y1::Int64, x2::Int64, y2::Int64)
    return sqrt(abs(x1 - x2) ^ 2 + abs(y1 - y2) ^2)
end 

# transition helper for reward 
function check_in_hot_spot(x::Int64, y::Int64, hot_spots::Vector{Vector{Int64}})
    min_dist = minimum(distance(x, y, point[1], point[2]) for point in hot_spots)
    random_num =  rand()
    if random_num <= 1/ (min_dist + 1)
        return true
    end 
    return false
end

function POMDPs.transition(mdp::taxi_world, state::taxi_world_state, action::Symbol)
    a = action
    x = state.x
    y = state.y
    temp = state.temp
    time = state.time

    prob_request = 0.3
    
    # If we've received a request, we do not transition 
    if state.received_request
        return SparseCat([taxi_world(x, y, temp, time, true)], [1.0])
    end
    hot_spots = []
    # determine hot spots based on weather 
    if temp == 3 || temp == 4 # warm  
        hot_spots = [[20,50], [40, 20], [50, 50], [70, 60],	[80, 70], [40, 90],	[80, 90], [90, 80]]
    else # cold
        hot_spots = [[100, 40], [20, 45], [30, 55], [40, 90], [60, 60], [80, 75], [90, 65], [70, 95]]
    end
    curr_dist_from_hs = minimum(distance(x, y, point[1], point[2]) for point in hot_spots)
    stuck_prob = 0
    if time == 3 || time == 4
        stuck_prob = 1.0 / (curr_dist_from_hs + 1)
    else 
        stuck_prob = 1.0 / (2 * (curr_dist_from_hs + 1))
    end
    

    neighbors = [
        taxi_world_state(x, y+1, temp, time, check_in_hot_spot(x, y+1, hot_spots)), # up
        taxi_world_state(x, y-1, temp, time, check_in_hot_spot(x, y-1, hot_spots)), # down
        taxi_world_state(x-1, y, temp, time, check_in_hot_spot(x-1, y, hot_spots)), # left
        taxi_world_state(x+1, y, temp, time, check_in_hot_spot(x+1, y, hot_spots)), # right
        taxi_world_state(x, y, temp, time, check_in_hot_spot(x, y, hot_spots)) #stay
        ] 
    
    targets = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :stay=>5) 
    target = targets[a]
    probability = fill(0.0, 5)

    if !inbounds(mdp, neighbors[target])
        # If would transition out of bounds, stay in
        # same cell with probability 1
        return SparseCat([taxi_world_state(x, y, temp, time, state.received_request)], [1.0])
    else
        probability[target] = 1 - stuck_prob
        probability[5] = stuck_prob
    end

    return SparseCat(neighbors, probability)
end;

function POMDPs.reward(mdp::taxi_world, state::taxi_world_state, action::Symbol, sp::taxi_world_state) 
    r = 0
    if state.received_request
        return 0.0
    end
    targets = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :stay=>5) 
    if state != sp
        r += mdp.reward_for_gas
    end
    if sp.received_request
        r += mdp.reward_for_request
    end
    return r
end;

POMDPs.discount(mdp::taxi_world) = mdp.discount_factor;

function POMDPs.actionindex(mdp::taxi_world, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    elseif act==:stay
        return 5
    end
    error("Invalid GridWorld action: $act")
end;

function POMDPs.stateindex(mdp::taxi_world, state::taxi_world_state)
    sd = Int(state.received_request + 1)
    ci = CartesianIndices((mdp.size_x, mdp.size_y, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

POMDPs.isterminal(mdp::taxi_world, s::taxi_world_state) = s.received_request

function get_action_index(act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    elseif act==:stay
        return 5
    end
end;



# TODO: COMMENT ME OUT!!!!!!!!!

 mdp = taxi_world() # create the taxi world MDP
 policy = RandomPolicy(mdp)
 data = Vector()
 push!(data, ("x", "y", "temp", "time", "received_request", "a", "r", "sp_x", "sp_y", "sp_temp", "sp_time", "sp_received_request")) # titles

for i in 1:10
     for x in 1:100
         for y in 1:100
             for time in 1:4
                 for temp in 1:4
                     POMDPs.initialstate(mdp::taxi_world) = Deterministic(taxi_world_state(x, y, time, temp, false))
                     for (s,a,r, sp) in stepthrough(mdp, policy, "s,a,r,sp", max_steps=100)
                         push!(data, (s.x, s.y, s.temp, s.time, s.received_request, get_action_index(a), r, sp.x, sp.y, sp.temp, sp.time, sp.received_request))
                         writedlm("train_dataset_new.txt", data)
                     end
                 end
             end
         end
     end
 end
 writedlm("train_dataset_new.txt", data)
