"""
Helper functions related to transition matrices and uniformization.

"""
from __future__ import division, print_function, absolute_import

#TODO move this to a more appropriate python package

import networkx as nx


def get_omega(total_rates, uniformization_factor):
    return uniformization_factor * max(total_rates.values())


def get_poisson_rates(total_rates, omega):
    return dict((s, omega - r) for s, r in total_rates.items())


def get_total_rates(Q_nx):
    """
    
    Parameters
    ----------
    Q_nx : directed networkx graph
        rate matrix

    Returns
    -------
    total_rates : dict
        map from state to total rate away from the state

    """
    total_rates = {}
    for sa in Q_nx:
        total_rate = None
        for sb in Q_nx[sa]:
            rate = Q_nx[sa][sb]['weight']
            if total_rate is None:
                total_rate = 0
            total_rate += rate
        if total_rate is not None:
            total_rates[sa] = total_rate
    return total_rates


def get_uniformized_P_nx(Q_nx, omega):
    """

    Parameters
    ----------
    Q_nx : directed networkx graph
        rate matrix
    omega : float
        uniformization rate

    Returns
    -------
    P_nx : directed networkx graph
        transition probability matrix

    """
    total_rates = get_total_rates(Q_nx)
    P_nx = nx.DiGraph()
    for sa in Q_nx:
        total_rate = total_rates.get(sa, 0)

        # define the self-transition probability
        weight = 1.0 - total_rate / omega
        P_nx.add_edge(sa, sa, weight=weight)

        # define probabilities of transitions to other states
        for sb in Q_nx[sa]:
            weight = Q_nx[sa][sb]['weight'] / omega
            P_nx.add_edge(sa, sb, weight=weight)

    return P_nx


def get_identity_P_nx(states):
    P_nx = nx.DiGraph()
    for s in states:
        P_nx.add_edge(s, s, weight=1)
    return P_nx


def get_node_to_tm(T, root, node_to_blen):
    node_to_tm = {root : 0}
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        node_to_tm[vb] = node_to_tm[va] + node_to_blen[edge]
    return node_to_tm
