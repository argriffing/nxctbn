"""
This toy model is described in raoteh/examples/codon2x3.

mini glossary
tol -- tolerance
traj -- trajectory
fg -- foreground track
bg -- background track

The tree is rooted and edges are directed.
For each substate track, each permanent node maps to a list of events.
Each event is a handle mapping to some event info giving the
time of the event along the branch and the nature of the transition,
if any, associated with the event.

We can use the jargon that 'events' are associated with
locations in the tree defined by a directed edge
and a distance along that edge.
Events will usually be associated with a state transition,
but 'incomplete events' will not have such an association.

The process is separated into multiple 'tracks' -- a primary process
track and one track for each of the tolerance processes.
The track trajectories are not independent of each other.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from functools import partial
from itertools import product
import warnings

import networkx as nx
import numpy as np
import scipy.linalg

import nxmctree
from nxmctree.sampling import sample_history

from util import get_node_to_tm, get_total_rates
from trajectory import Trajectory, Event


RATE_ON = 1.0
RATE_OFF = 1.0

#NTOLS = 3

#alpha = RATE_ON / (RATE_ON + RATE_OFF)
#t = 1 / NTOLS

#P_ON = t * 1 + (1-t) * alpha
#P_OFF = t * 0 + (1-t) * (1-alpha)

P_ON = RATE_ON / (RATE_ON + RATE_OFF)
P_OFF = RATE_OFF / (RATE_ON + RATE_OFF)


#TODO adjust the poisson rate correctly for each segment
# using the background context
# adjust the transition matrices accordingly

#TODO use 'segment' vs. 'edge' jargon


###############################################################################
# Primary track and blink track initialization.



def init_blink_history(T, track):
    """
    Initial blink history is True where consistent with the data.

    """
    for v in T:
        track.history[v] = (True in track.data[v])


def init_complete_blink_events(T, node_to_tm, track):
    """
    Init blink track.

    """
    for edge in T.edges():
        va, vb = edge
        sa = track.history[va]
        sb = track.history[vb]
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        tma = edge_tma + blen * np.random.uniform(0, 1/3)
        tmb = edge_tma + blen * np.random.uniform(2/3, 1)
        eva = Event(track=track, tm=tma, sa=sa, sb=True)
        evb = Event(track=track, tm=tmb, sa=True, sb=sb)
        track.events[edge] = [eva, evb]


def init_incomplete_primary_events(T, node_to_tm, primary_track, diameter):
    """
    Parameters
    ----------
    T : nx tree
        tree
    node_to_tm : dict
        maps nodes to times
    primary_track : Trajectory
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    """
    for edge in T.edges():
        va, vb = edge
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        times = edge_tma + blen * np.random.uniform(
                low=1/3, high=2/3, size=diameter-1)
        events = [Event(track=primary_track, tm=tm) for tm in times]
        primary_track.events[edge] = events


###############################################################################
# Classes and functions for steps of Rao Teh iteration.


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
        fset : set, optional
            Set of foreground state restrictions or None.
            None is interpreted as no restriction rather than
            lack of feasible states.
        transition : triple, optional
            A transition like (trajectory_name, sa, sb) or None.
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


def do_nothing(state):
    """
    Helper function as a placeholder callback.

    """
    pass


def set_or_confirm_history_state(node_to_state, node, state):
    """
    Helper function for updating history within a trajectory.

    """
    if node_to_state.get(node, None) not in (state, None):
        raise Exception('found a history incompatibility')
    node_to_state[node] = state


def get_edge_tree(T, root):
    """
    Nodes in the edge tree are edges in the original tree.

    The new tree will have a node (None, root) which does not correspond
    to any edge in the original tree.

    """
    dual_root = (None, root)
    T_dual = nx.DiGraph()
    if not T:
        return T_dual, dual_root
    for c in T[root]:
        T_dual.add_edge(dual_root, (root, c))
    for v in T:
        for c in T[v]:
            for g in T[c]:
                T_dual.add_edge((v, c), (c, g))
    return T_dual, dual_root


def get_blink_dwell_times(T, node_to_tm, blink_tracks):
    dwell_off = 0
    dwell_on = 0
    for edge in T.edges():
        va, vb = edge
        tma = node_to_tm[va]
        tmb = node_to_tm[vb]

        # construct unordered collection of events
        events = []
        for track in blink_tracks:
            events.extend(track.events[edge])

        # construct ordered sequence of events and endpoints
        seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in sorted(events)]
        info_a = (tma, None, None, None)
        info_b = (tmb, None, None, None)
        seq = [info_a] + seq + [info_b]

        # initialize the state of each blining track along the edge
        track_to_state = {}
        for track in blink_tracks:
            track_to_state[track.name] = track.history[va]

        # compute dwell times associated with each trajectory on each segment
        for segment in zip(seq[:-1], seq[1:]):
            info_a, info_b = segment
            tma, tracka, saa, sba = info_a
            tmb, trackb, sab, sbb = info_b
            blen = tmb - tma

            # Keep the state of each track up to date.
            if tracka is not None:
                tm, track, sa, sb = info_a
                name = tracka.name
                if track_to_state[name] != sa:
                    raise Exception('incompatible transition: '
                            'current state on track %s is %s '
                            'but encountered a transition event from '
                            'state %s to state %s' % (
                                name, track_to_state[name], sa, sb))
                track_to_state[name] = sb

            # update the dwell times
            for track in blink_tracks:
                state = track_to_state[track.name]
                if state == False:
                    dwell_off += blen
                elif state == True:
                    dwell_on += blen
                else:
                    raise Exception

    # return the dwell times
    return dwell_off, dwell_on



def sample_poisson_events(T, node_to_tm, fg_track, bg_tracks, bg_to_fg_fset):
    """
    Sample poisson events on the tree.

    The poisson rate is piecewise homogeneous on the tree
    and depends not only on the foreground track,
    but also on the background tracks.

    """
    P_nx = fg_track.P_nx
    P_nx_identity = fg_track.P_nx_identity

    for edge in T.edges():
        va, vb = edge
        tma = node_to_tm[va]
        tmb = node_to_tm[vb]

        events = []
        events.extend(fg_track.events[edge])
        for bg_track in bg_tracks:
            events.extend(bg_track.events[edge])

        # Construct the meta nodes corresponding to sorted events.
        # No times should coincide.
        seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in sorted(events)]
        info_a = (tma, None, None, None)
        info_b = (tmb, None, None, None)
        seq = [info_a] + seq + [info_b]

        # Initialize foreground state and background states
        # at the beginning of the edge.
        track_to_state = {}
        for bg_track in bg_tracks:
            track_to_state[bg_track.name] = bg_track.history[va]
        track_to_state[fg_track.name] = fg_track.history[va]

        # Iterate over segments of the edge.
        # Within each segment the foreground and background tracks
        # maintain the same state, and therefore the rate
        # of new foreground poisson events is constant within the segment.
        poisson_events = []
        for segment in zip(seq[:-1], seq[1:]):
            info_a, info_b = segment
            tma, tracka, saa, sba = info_a
            tmb, trackb, sab, sbb = info_b
            blen = tmb - tma

            # Keep the state of each track up to date.
            if tracka is not None:
                tm, track, sa, sb = info_a
                name = tracka.name
                if track_to_state[name] != sa:
                    raise Exception('incompatible transition: '
                            'current state on track %s is %s '
                            'but encountered a transition event from '
                            'state %s to state %s' % (
                                name, track_to_state[name], sa, sb))
                track_to_state[name] = sb

            # Use the foreground and background track states
            # to define the poisson rate that is homogeneous on this segment.
            rate = 0
            fg_sa = track_to_state[fg_track.name]
            for fg_sb in fg_track.Q_nx[fg_sa]:
                tolerated = True
                for bg_track in bg_tracks:
                    bg_state = track_to_state[bg_track.name]
                    fset = bg_to_fg_fset[bg_track.name][bg_state]
                    if fg_sb not in fset:
                        tolerated = False
                if tolerated:
                    rate += fg_track.Q_nx[fg_sa][fg_sb]['weight']
            #poisson_rate = fg_track.omega - rate
            #TODO hack
            poisson_rate = fg_track.poisson_rates[fg_sa]

            # Sample some poisson events on the segment.
            nevents = np.random.poisson(poisson_rate * blen)
            times = np.random.uniform(low=tma, high=tmb, size=nevents)
            for tm in times:
                ev = Event(track=fg_track, tm=tm)
                poisson_events.append(ev)

        # Add the poisson events into the list of foreground
        # track events for this edge.
        fg_track.events[edge].extend(poisson_events)


def foo():
    """
    Moved out of sample_transitions

    """
    # Use the states of the background tracks,
    # together with fsets of the two meta nodes if applicable,
    # to define the set of feasible foreground states at this segment.
    fsets = []
    for m in segment:
        if m.fset is not None:
            fsets.append(m.fset)
    for name, state in bg_track_to_state.items():
        fsets.append(bg_to_fg_fset[name][state])
    fset = set.intersection(*fsets)


def sample_transitions(T, root, node_to_tm, primary_to_tol,
        fg_track, bg_tracks, bg_to_fg_fset, Q_meta):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    P_nx = fg_track.P_nx
    P_nx_identity = fg_track.P_nx_identity

    # Construct a meta node for each structural node.
    node_to_meta = {}
    #print('building meta nodes corresponding to structural nodes')
    for v in T:
        f = partial(set_or_confirm_history_state, fg_track.history, v)
        fset = fg_track.data[v]
        m = MetaNode(P_nx=P_nx_identity, set_sa=f, set_sb=f, fset=fset,
                tm=node_to_tm[v])
        node_to_meta[v] = m
        #print('adding meta node', v, m)

    # Define the meta node corresponding to the root.
    mroot = node_to_meta[root]

    # Build the tree whose vertices are meta nodes,
    # and map edges of this tree to sets of feasible foreground states,
    # accounting for data at structural nodes and background context
    # along edge segments.
    #
    # Also create the map from edges of this tree
    # to sets of primary states not directly contradicted by data or context.
    #
    meta_node_tree = nx.DiGraph()
    node_to_data_lmap = dict()
    for edge in T.edges():
        va, vb = edge

        # Sequence meta nodes from three sources:
        # the two structural endpoint nodes,
        # the nodes representing transitions in background tracks,
        # and nodes representing transitions in the foreground track.
        # Note that meta nodes are not meaningfully sortable,
        # but events are sortable.
        events = []
        events.extend(fg_track.events[edge])
        for bg_track in bg_tracks:
            events.extend(bg_track.events[edge])

        # Construct the meta nodes corresponding to sorted events.
        seq = []
        for ev in sorted(events):
            if ev.track is fg_track:
                m = MetaNode(P_nx=P_nx, set_sa=ev.init_sa, set_sb=ev.init_sb,
                        tm=ev.tm)
            else:
                m = MetaNode(P_nx=P_nx_identity,
                        set_sa=do_nothing, set_sb=do_nothing,
                        transition=(ev.track.name, ev.sa, ev.sb),
                        tm=ev.tm)
            seq.append(m)
        ma = node_to_meta[va]
        mb = node_to_meta[vb]
        seq = [ma] + seq + [mb]

        # Initialize background states at the beginning of the edge.
        bg_track_to_state = {}
        for bg_track in bg_tracks:
            bg_track_to_state[bg_track.name] = bg_track.history[va]

        # Add segments of the edge as edges of the meta node tree.
        # Track the state of each background track at each segment.
        #print('processing edge', va, vb)
        for segment in zip(seq[:-1], seq[1:]):
            ma, mb = segment

            # Keep the state of each background track up to date.
            if ma.transition is not None:
                name, sa, sb = ma.transition
                if bg_track_to_state[name] != sa:
                    raise Exception('incompatible transition: '
                            'current state on track %s is %s '
                            'but encountered a transition event from '
                            'state %s to state %s' % (
                                name, bg_track_to_state[name], sa, sb))
                bg_track_to_state[name] = sb

            # Get the set of states allowed by data and background interaction.
            fsets = []
            for m in segment:
                if m.fset is not None:
                    fsets.append(m.fset)
            for name, state in bg_track_to_state.items():
                fsets.append(bg_to_fg_fset[name][state])
            fg_allowed = set.intersection(*fsets)
            # For each possible foreground state,
            # use the states of the background tracks and the data
            # to determine foreground feasibility
            # and possibly a multiplicative rate penalty.
            lmap = dict()
            if len(bg_tracks) > 1:
                # Foreground is the primary track.
                # Use the states of the background blinking tracks,
                # together with fsets of the two meta nodes if applicable,
                # to define the set of feasible foreground states
                # at this segment.
                lmap = dict((s, 1) for s in fg_allowed)
            else:
                # Foreground is a blinking track.
                # The lmap has nontrivial penalties
                # depending on both the background (primary) track state
                # and the proposed foreground blink state.
                pri_track = bg_tracks[0]
                pri_state = bg_track_to_state[pri_track.name]
                if False in fg_allowed:
                    lmap[False] = 1
                # The blink state choice of True should be penalized
                # according to the sum of rates from the current
                # primary state to primary states controlled by
                # the proposed foreground track.
                #if False:
                if True in fg_allowed:
                    #if primary_to_tol[pri_state] == fg_track.name:
                        #lmap[True] = 1
                    if Q_meta.has_edge(pri_state, fg_track.name):
                        #print('effectively disallowing some blinked-on states')
                        rate_sum = Q_meta[pri_state][fg_track.name]['weight']
                        amount = rate_sum * (mb.tm - ma.tm)
                        lmap[True] = np.exp(-amount)
                    else:
                        lmap[True] = 1

            # Map the segment to the lmap.
            # Segments will be nodes of the tree whose history will be sampled.
            node_to_data_lmap[segment] = lmap

            # Add the meta node to the meta node tree.
            #print('adding segment', ma, mb)
            meta_node_tree.add_edge(ma, mb)

    # Build the tree whose vertices are edges of the meta node tree.
    meta_edge_tree, meta_edge_root = get_edge_tree(meta_node_tree, mroot)
    #print('size of meta_edge_tree:')
    #print(len(meta_edge_tree))
    #print()
    #print('meta edge root:')
    #print(meta_edge_root)
    #print()

    # Create the map from edges of the meta edge tree
    # to primary state transition matrices.
    edge_to_P = {}
    for pair in meta_edge_tree.edges():
        (ma, mb), (mb2, mc) = pair
        if mb != mb2:
            raise Exception('incompatibly constructed meta edge tree')
        edge_to_P[pair] = mb.P_nx

    # Use nxmctree to sample a history on the meta edge tree.
    root_data_fset = fg_track.data[root]
    #print(root_data_fset)
    #print(node_to_data_lmap)
    node_to_data_lmap[meta_edge_root] = dict((s, 1) for s in root_data_fset)
    meta_edge_to_sampled_state = sample_history(
            meta_edge_tree, edge_to_P, meta_edge_root,
            fg_track.prior_root_distn, node_to_data_lmap)
    #print('size of sampled history:')
    #print(len(meta_edge_to_sampled_state))
    #print()

    # Use the sampled history to update the primary history at structural nodes
    # and to update the primary event transitions.
    for meta_edge in meta_edge_tree:
        ma, mb = meta_edge
        state = meta_edge_to_sampled_state[meta_edge]
        if ma is not None:
            ma.set_sb(state)
        if mb is not None:
            mb.set_sa(state)



###############################################################################
# Main Rao-Teh-Gibbs sampling function.


def blinking_model_rao_teh(
        T, root, node_to_tm, primary_to_tol,
        Q_primary, Q_blink, Q_meta,
        primary_track, tolerance_tracks, interaction_map):
    """

    Parameters
    ----------
    T : x
        x
    root : x
        x
    node_to_tm : x
        x
    primary_to_tol : x
        x
    Q_primary : x
        x
    Q_blink : x
        x
    primary_track : hashable
        label of the primary track
    tolerance_tracks : collection of hashables
        labels of tolerance tracks
    interaction_map : dict
        x

    """
    # Initialize blink history and events.
    for track in tolerance_tracks:
        init_blink_history(T, track)
        init_complete_blink_events(T, node_to_tm, track)
        track.remove_self_transitions()

    # Initialize the primary trajectory with many incomplete events.
    diameter = 4
    init_incomplete_primary_events(T, node_to_tm, primary_track, diameter)
    #
    # Sample the state of the primary track.
    sample_transitions(T, root, node_to_tm, primary_to_tol,
            primary_track, tolerance_tracks, interaction_map['P'], Q_meta)
    #
    # Remove self-transition events from the primary track.
    primary_track.remove_self_transitions()

    # Outer loop of the Rao-Teh-Gibbs sampler.
    while True:

        # Update the primary track.
        sample_poisson_events(T, node_to_tm,
                primary_track, tolerance_tracks, interaction_map['P'])
        primary_track.clear_state_labels()
        sample_transitions(T, root, node_to_tm, primary_to_tol,
                primary_track, tolerance_tracks, interaction_map['P'], Q_meta)
        primary_track.remove_self_transitions()

        # Update each blinking track.
        for track in tolerance_tracks:
            name = track.name
            #print('adding poisson events for track', name)
            sample_poisson_events(T, node_to_tm,
                    track, [primary_track], interaction_map[name])
            #track.add_poisson_events(T, node_to_tm)
            #print('clearing state labels for track', name)
            track.clear_state_labels()
            #print('sampling state transitions for track', name)
            sample_transitions(T, root, node_to_tm, primary_to_tol,
                    track, [primary_track], interaction_map[name], Q_meta)
            #print('removing self transitions for track', name)
            track.remove_self_transitions()

        """
        # Summarize the sample.
        expected_on = 0
        expected_off = 0
        for track in tolerance_tracks:
            for edge in T.edges():
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if transition == (False, True):
                        expected_on += 1
                    elif transition == (True, False):
                        expected_off += 1

        yield expected_on, expected_off
        """

        # Yield the track states.
        yield primary_track, tolerance_tracks



###############################################################################
# Model specification code etc.


def get_Q_primary():
    # this is like a symmetric codon rate matrix
    rate = 1
    Q_primary = nx.DiGraph()
    Q_primary.add_weighted_edges_from((
        (0, 1, rate),
        (0, 2, rate),
        (1, 0, rate),
        (1, 3, rate),
        (2, 0, rate),
        (2, 3, rate),
        (2, 4, rate),
        (3, 1, rate),
        (3, 2, rate),
        (3, 5, rate),
        (4, 2, rate),
        (4, 5, rate),
        (5, 3, rate),
        (5, 4, rate),
        ))
    return Q_primary


def get_Q_blink():
    Q_blink = nx.DiGraph()
    Q_blink.add_weighted_edges_from((
        (False, True, RATE_ON),
        (True, False, RATE_OFF),
        ))
    return Q_blink


def get_primary_to_tol():
    """
    Return a map from primary state to tolerance track name.

    This is like a genetic code mapping codons to amino acids.

    """
    primary_to_tol = {
            0 : 'T0',
            1 : 'T0',
            2 : 'T1',
            3 : 'T1',
            4 : 'T2',
            5 : 'T2',
            }
    return primary_to_tol


def get_Q_meta(Q_primary, primary_to_tol):
    """
    Return a DiGraph of rates from primary states into sets of states.

    """
    Q_meta = nx.DiGraph()
    for primary_sa, primary_sb in Q_primary.edges():
        rate = Q_primary[primary_sa][primary_sb]['weight']
        tol_sb = primary_to_tol[primary_sb]
        if not Q_meta.has_edge(primary_sa, tol_sb):
            Q_meta.add_edge(primary_sa, tol_sb, weight=rate)
        else:
            Q_meta[primary_sa][tol_sb]['weight'] += rate
    return Q_meta


def get_T_and_root():
    # rooted tree, deliberately without branch lengths
    T = nx.DiGraph()
    T.add_edges_from([
        ('N1', 'N0'),
        ('N1', 'N2'),
        ('N1', 'N5'),
        ('N2', 'N3'),
        ('N2', 'N4'),
        ])
    return T, 'N1'


def get_edge_to_blen():
    edge_to_blen = {
            ('N1', 'N0') : 0.5,
            ('N1', 'N2') : 0.5,
            ('N1', 'N5') : 0.5,
            ('N2', 'N3') : 0.5,
            ('N2', 'N4') : 0.5,
            }
    return edge_to_blen


def hamming_distance(va, vb):
    return sum(1 for a, b in zip(va, vb) if a != b)


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False


def run(primary_to_tol, interaction_map, track_to_node_to_data_fset):

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the primary rate matrix.
    Q_primary = get_Q_primary()

    # Define the prior primary state distribution.
    #TODO do not use hardcoded uniform distribution
    nprimary = 6
    primary_distn = dict((s, 1/nprimary) for s in range(nprimary))

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary.edges():
        p = primary_distn[sa]
        rate = Q_primary[sa][sb]['weight']
        expected_primary_rate += p * rate
    #
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print()
    #
    for sa, sb in Q_primary.edges():
        Q_primary[sa][sb]['weight'] /= expected_primary_rate

    # Define primary trajectory.
    primary_track = Trajectory(
            name='P', data=track_to_node_to_data_fset['P'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    Q_blink = get_Q_blink()

    # Define the prior blink state distribution.
    blink_distn = {False : P_OFF, True : P_ON}

    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Define tolerance process trajectories.
    tolerance_tracks = []
    for name in ('T0', 'T1', 'T2'):
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    k = 800
    #k = 400
    #k = 320
    #k = 200
    #k = 100
    nsamples = k * k
    burnin = nsamples // 10
    ncounted = 0
    total_dwell_off = 0
    total_dwell_on = 0
    for i, (pri_track, tol_tracks) in enumerate(blinking_model_rao_teh(
            T, root, node_to_tm, primary_to_tol,
            Q_primary, Q_blink, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled < burnin:
            continue
        # Summarize the trajectories.
        for edge in T.edges():
            va, vb = edge
            for track in tol_tracks:
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if ev.sa == ev.sb:
                        raise Exception('self-transitions should not remain')
                    if transition == (False, True):
                        va_vb_type_to_count[va, vb, 'on'] += 1
                    elif transition == (True, False):
                        va_vb_type_to_count[va, vb, 'off'] += 1
            for ev in pri_track.events[edge]:
                transition = (ev.sa, ev.sb)
                if ev.sa == ev.sb:
                    raise Exception('self-transitions should not remain')
                if primary_to_tol[ev.sa] == primary_to_tol[ev.sb]:
                    va_vb_type_to_count[va, vb, 'syn'] += 1
                else:
                    va_vb_type_to_count[va, vb, 'non'] += 1
        dwell_off, dwell_on = get_blink_dwell_times(T, node_to_tm, tol_tracks)
        total_dwell_off += dwell_off
        total_dwell_on += dwell_on
        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break

    # report infos
    print('burnin:', burnin)
    print('samples after burnin:', nsamples)
    for va_vb_type, count in sorted(va_vb_type_to_count.items()):
        va, vb, s = va_vb_type
        print(va, '->', vb, s, ':', count / nsamples)
    print('dwell off:', total_dwell_off / nsamples)
    print('dwell on :', total_dwell_on / nsamples)


def main():

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()

    # Define track interactions.
    # This is analogous to the creation of the compound rate matrices.
    interaction_map = {
            'P' : {
                'T0' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {2, 3, 4, 5},
                    },
                'T1' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 4, 5},
                    },
                'T2' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 2, 3},
                    }
                },
            'T0' : {
                'P' : {
                    0 : {True},
                    1 : {True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T1' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {True},
                    3 : {True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T2' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {True},
                    5 : {True},
                    }
                }
            }


    # No data.
    print ('expectations given no alignment or disease data')
    print()
    data = {
            'P' : {
                'N0' : {0, 1, 2, 3, 4, 5},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {0, 1, 2, 3, 4, 5},
                'N4' : {0, 1, 2, 3, 4, 5},
                'N5' : {0, 1, 2, 3, 4, 5},
                },
            'T0' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            }
    run(primary_to_tol, interaction_map, data)
    print()


    # Alignment data only.
    print ('expectations given only alignment data but not disease data')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()


    # Alignment and disease data.
    print ('expectations given alignment and disease data')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()

    # Alignment and fully observed disease data.
    print ('expectations given alignment and fully observed disease data')
    print ('(all leaf disease states which were previously considered to be')
    print ('unobserved are now considered to be tolerated (blinked on))')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()


main()

