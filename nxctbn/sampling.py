"""
Sample from the CTBN using the method of Rao and Teh.

The model will be represented by a callback function or maybe an object
that supplies a rate matrix (or poisson rates together with
a uniformized transition matrix?) for each combination of a foreground
CTBN node and its parent CTBN nodes.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


def get_omega(rates, uniformization_factor):
    return uniformization_factor * max(rates.values())


def get_poisson_rates(rates, omega):
    return dict((s, omega - r) for s, r in rates.items())


def get_rates(Q):
    """
    
    Parameters
    ----------
    Q : directed networkx graph
        rate matrix

    Returns
    -------
    rates : dict
        map from state to total rate away from the state

    """
    rates = {}
    for sa in Q:
        rate = None
        for sb in Q[sa]:
            r = Q[sa][sb]['weight']
            if rate is None:
                rate = 0
            rate += r
        if rate is not None:
            rates[sa] = rate
    return rates


def get_P(Q, rates, omega):
    """

    Parameters
    ----------
    Q : directed networkx graph
        rate matrix
    rates : dict
        map from state to total rate away from the state
    omega : float
        uniformization rate

    Returns
    -------
    P : directed networkx graph
        transition probability matrix

    """
    P = nx.DiGraph()
    for sa in Q:

        # define the self-transition probability
        rate = rages.get(sa, 0)
        p = 1.0 - rate / omega
        P.add_edge(sa, sa, weight=p)

        # define probabilities of transitions to other states
        for sb in Q[sa]:
            rate = Q[sa][sb]['weight']
            p = rate / omega
            P.add_edge(sa, sb, weight=p)

    return P


class MetaNode(object):
    """
    This is hashable so it can be a node in a networkx graph.

    """
    def __init__(self, P_nx=None,
            set_sa=None, set_sb=None, fset=None, transition=None, tm=None):
        """

        Parameters
        ----------
        P_nx : nx transition matrix, optional
            the node is associated with this transition matrix
        set_sa : callback, optional
            report the sampled initial state
        set_sb : callback, optional
            report the sampled final state
        lmap : dict, optional
            data likelihood conditional at a structural node
            as a function of the unknown foreground state
            and the fixed background states.
        transition : triple, optional
            A transition like (track_name, sa, sb) or None.
            None is interpreted as absence of background transition.
        tm : float
            time elapsed since the root

        """
        self.P_nx = P_nx
        self.set_sa = set_sa
        self.set_sb = set_sb
        self.fset = fset
        self.transition = transition
        self.tm = tm

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


def inner_gibbs(ctbn, name_to_track, track_name):
    """

    Parameters
    ----------
    ctbn : DiGraph
        Directed CTBN graph.
        Every CTBN node should be a vertex in this graph.
        Every direct dependence between CTBN node trajectories should
        be represented by an edge u->v if CTBN node v directly depends
        on node u.
    name_to_track : dict
        Map from track name to track object.
    track_name : hashable
        Name of the foreground track.
        The trajectory of this track will be updated according to the CTBN
        sampling scheme of Rao and Teh.

    """
    ps = ctbn.predecessors(track_name)
    cs = ctbn.successors(track_name)


def gen_trajectories(ctbn, name_to_track):
    """

    Parameters
    ----------
    ctbn : DiGraph
        Directed CTBN graph.
        Every CTBN node should be a vertex in this graph.
        Every direct dependence between CTBN node trajectories should
        be represented by an edge u->v if CTBN node v directly depends
        on node u.
    name_to_track : dict
        Map from track name to track object.

    """
    # Outer loop of the Gibbs sampler.
    while True:

        # Inner loop of the Gibbs sampler.
        for cnode in ctbn:
            inner_gibbs(ctbn, name_to_track, cnode)

        # The yielded dict is not really necessary because it is the
        # same as the input dict which is modified in-place.
        yield name_to_track

