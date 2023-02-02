import sys
import networkx as nx
import pandas as pd
import math
import random
import numpy as np


def create_blank_graph():
    G = nx.Graph()

    # create nodes
    golden_gate_park = "golden_gate_park"
    sunset = "sunset"
    bayview = "bayview"
    mission = "mission"
    fishermans_wharf = "fishermans_wharf"
    nodes_list = [golden_gate_park, sunset, bayview, mission, fishermans_wharf]
    G.add_nodes_from(nodes_list)

    # add edges with weights
    G.add_edge(fishermans_wharf, golden_gate_park, weight=3)
    G.add_edge(golden_gate_park, sunset, weight=1)
    G.add_edge(sunset, bayview, weight=2)
    G.add_edge(bayview, mission, weight=1)
    G.add_edge(mission, fishermans_wharf, weight=4)

    return G


def simulate_data():
    df = pd.DataFrame(columns = ['time', 'weather', 'weekend', 'holiday', 'node', 'value'])

    # create probabilities for each variable
    days = 100
    time = {0: 0.5, 6: 0.1, 12: 0.3, 18: 0.9}  # hours after midnight
    weather = {50: 0.8, 60: 0.4, 70: 0.4, 80: 0.5, 90: 0.8}  # degrees in F
    weekend = {0: 0.5, 1: 0.4}  # weekday, weekend
    holiday = {0: 0.2, 1: 0.3}
    node = {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8}

    # determine product
    product = 0
    for t in time:
        for w in weather:
            for we in weekend:
                for h in holiday:
                    for n in node:
                        product += t * w * we * h * n

    # simulate data
    for day in range(days):
        t = np.random.choice(list(time.keys()))
        w = np.random.choice(list(weather.keys()))
        we = np.random.choice(list(weekend.keys()))
        h = np.random.choice(list(holiday.keys()))
        n = np.random.choice(list(node.keys()))

        new_data = dict()
        new_data['time'] = t
        new_data['weather'] = w
        new_data['weekend'] = we
        new_data['holiday'] = h
        new_data['node'] = n

        prob = (time[t] * weather[w] * weekend[we] * holiday[h] * node[n]) / product
        #TODO: the probabilities are tiny so value is always 0
        value = np.random.binomial(1, prob)

        new_data['value'] = value
        df.loc[len(df.index)] = new_data

    return df


def main():
    G = create_blank_graph()
    df = simulate_data()

if __name__ == '__main__':
    main()