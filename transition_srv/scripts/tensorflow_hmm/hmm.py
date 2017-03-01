import tensorflow as tf
import numpy as np


class HMM(object):
    """
    A class for Hidden Markov Models.

    The model attributes are:
    - K :: the number of states
    - P :: the K by K transition matrix (from state i to state j,
        (i, j) in [1..K])
    - p0 :: the initial distribution (defaults to starting in state 0)
    """

    def __init__(self, P, p0=None):
        self.K = P.shape[0]

        self.P = P
        self.logP = np.log(self.P)

        if p0 is None:
            self.p0 = np.ones(self.K)
            self.p0 /= sum(self.p0)
        elif len(p0) != self.K:
            raise ValueError(
                'dimensions of p0 {} must match P[0] {}'.format(
                    p0.shape, P.shape[0]))
        else:
            self.p0 = p0
        self.logp0 = np.log(self.p0)


class HMMNumpy(HMM):

    def forward_backward(self, y):
        # set up
        nT = y.shape[0]
        posterior = np.zeros((nT, self.K))
        forward = np.zeros((nT + 1, self.K))
        backward = np.zeros((nT + 1, self.K))

        # forward pass
        forward[0, :] = 1.0 / self.K
        for t in range(nT):
            tmp = np.multiply(
                np.matmul(forward[t, :], self.P),
                y[t]
            )

            forward[t + 1, :] = tmp / np.sum(tmp)

        # backward pass
        backward[-1, :] = 1.0 / self.K
        for t in range(nT, 0, -1):
            tmp = np.matmul(
                np.matmul(
                    self.P, np.diag(y[t - 1])
                ),
                backward[t, :].transpose()
            ).transpose()

            backward[t - 1, :] = tmp / np.sum(tmp)

        # remove initial/final probabilities
        forward = forward[1:, :]
        backward = backward[:-1, :]

        # combine and normalize
        posterior = np.array(forward) * np.array(backward)
        # [:,None] expands sum to be correct size
        posterior = posterior / np.sum(posterior, 1)[:, None]

        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        tmpMat = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                tmpMat[i, j] = scores[i] + self.logP[i, j]
        return tmpMat

    def viterbi_decode(self, y):
        y = np.array(y)

        nT = y.shape[0]

        pathStates = np.zeros((nT, self.K), dtype=np.int)
        pathScores = np.zeros((nT, self.K))

        # initialize
        pathScores[0] = self.logp0 + np.log(y[0])

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])

            # the inferred state
            pathStates[t + 1] = np.argmax(tmpMat, 0)
            pathScores[t + 1] = np.max(tmpMat, 0) + np.log(yy)

        # now backtrack viterbi to find states
        s = np.zeros(nT, dtype=np.int)
        s[-1] = np.argmax(pathScores[-1])
        for t in range(nT - 1, 0, -1):
            s[t - 1] = pathStates[t, s[t]]

        return s, pathScores


class HMMTensorflow(HMM):

    def forward_backward(self, y):
        """
        runs forward backward algorithm on state probabilities y

        Arguments
        ---------
        y : np.array : shape (T, K) where T is number of timesteps and
            K is the number of states

        Returns
        -------
        (posterior, forward, backward)
        posterior : list of length T of tensorflow graph nodes representing
            the posterior probability of each state at each time step
        forward : list of length T of tensorflow graph nodes representing
            the forward probability of each state at each time step
        backward : list of length T of tensorflow graph nodes representing
            the backward probability of each state at each time step
        """
        # set up
        nT = y.shape[0]

        posterior = np.zeros((nT, self.K))
        forward = []
        backward = np.zeros((nT + 1, self.K))

        # forward pass
        forward.append(
            tf.ones((1, self.K), dtype=tf.float64) * (1.0 / self.K)
        )
        for t in range(nT):
            # NOTE: np.matrix expands forward[t, :] into 2d and causes * to be
            # matrix multiplies instead of element wise that an array would be
            tmp = tf.mul(
                tf.matmul(forward[t], self.P),
                y[t]
            )

            forward.append(tmp / tf.reduce_sum(tmp))

        # backward pass
        backward = [None] * (nT + 1)
        backward[-1] = tf.ones((1, self.K), dtype=tf.float64) * (1.0 / self.K)
        for t in range(nT, 0, -1):
            tmp = tf.transpose(
                tf.matmul(
                    tf.matmul(self.P, tf.diag(y[t - 1])),
                    tf.transpose(backward[t])
                )
            )
            backward[t - 1] = tmp / tf.reduce_sum(tmp)

        # remove initial/final probabilities
        forward = forward[1:]
        backward = backward[:-1]

        # combine and normalize
        posterior = [f * b for f, b in zip(forward, backward)]
        posterior = [p / tf.reduce_sum(p) for p in posterior]

        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        # first convert scores into shape [K, 1]
        # then concatenate K of them into shape [K, K]
        expanded_scores = tf.concat(
            1, [tf.expand_dims(scores, 1)] * self.K
        )
        return expanded_scores + self.logP

    def viterbi_decode(self, y):
        """
        Runs viterbi decode on state probabilies y.

        Arguments
        ---------
        y : np.array : shape (T, K) where T is number of timesteps and
            K is the number of states

        Returns
        -------
        (s, pathScores)
        s : list of length T of tensorflow ints : represents the most likely
            state at each time step.
        pathScores : list of length T of tensorflow tensor of length K
            each value at (t, k) is the log likliehood score in state k at
            time t.  sum(pathScores[t, :]) will not necessary == 1
        """
        y = np.asarray(y)
        if len(y.shape) != 2:
            raise ValueError((
                'y should be 2d of shape (nT, {}).  Found {}'
            ).format(self.K, y.shape))

        if y.shape[1] != self.K:
            raise ValueError((
                'y has an invalid shape.  first dimension is time and second '
                'is K.  Expected K for this model is {}, found {}.'
            ).format(self.K, y.shape[1]))

        nT = y.shape[0]

        # pathStates and pathScores wil be of type tf.Tensor.  They
        # are lists since tensorflow doesn't allow indexing, and the
        # list and order are only really necessary to build the unrolled
        # graph.  We never do any computation across all of time at once
        pathStates = []
        pathScores = []

        # initialize
        pathStates.append(None)
        pathScores.append(self.logp0 + np.log(y[0]))

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])

            # the inferred state
            pathStates.append(tf.argmax(tmpMat, 0))
            pathScores.append(tf.reduce_max(tmpMat, 0) + np.log(yy))

        # now backtrack viterbi to find states
        s = [0] * nT
        s[-1] = tf.argmax(pathScores[-1], 0)
        for t in range(nT - 1, 0, -1):
            s[t - 1] = tf.gather(pathStates[t], s[t])

        return s, pathScores
