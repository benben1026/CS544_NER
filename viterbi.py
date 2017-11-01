import numpy as np
import sys

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    best = [[0.0 for j in xrange(N)] for i in xrange(L)]
    bt = [[-1 for j in xrange(N)] for i in xrange(L)]
    for i in xrange(L):
        best[i][0] = start_scores[i] + emission_scores[0, i]
    for j in xrange(1, N):
        for i_c in xrange(L):
            max_score = -sys.maxint - 1
            for i_p in xrange(L):
                s = best[i_p][j - 1] + trans_scores[i_p][i_c] + emission_scores[j][i_c]
                if s > max_score:
                    max_score = s
                    bt[i_c][j] = i_p
            best[i_c][j] = max_score
    for i in xrange(L):
        best[i][-1] = best[i][-1] + end_scores[i]

    score = -sys.maxint - 1
    index = -1
    for i in xrange(L):
        if best[i][-1] > score:
            index = i
            score = best[i][-1]
    j = N - 1
    y = []
    while index != -1:
        y = [index] + y
        index = bt[index][j]
        j -= 1

    return (score, y)
