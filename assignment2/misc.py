##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    ep = sqrt(6) / sqrt(m + n)
    A0 = random.uniform(-ep, ep, (m, n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
