"""
Define parent-dependent transition rate matrices of nodes in a toy CTBN.

"""
#TODO move to an examples directory


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


def get_interactions():
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
