# ############################################################################
# optim.py
# ========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Optimization Tools.
"""

import logging
import pathlib
import time

import numpy as np


class DataSet:
    """
    Dataset of samples.

    Contains all information relative to a dataset.
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            (N_sample, N_data) encoded samples. (real-valued)
        """
        if not (data.ndim == 2):
            raise ValueError('Parameter[data] is ill-formed.')
        N_sample = data.shape[0]
        self._data = data

    def __len__(self):
        """
        Returns
        -------
        N_sample : int
            Number of samples in the dataset.
        """
        N_sample = self._data.shape[0]
        return N_sample

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : int / slice() / :py:class:`~numpy.ndarray`

        Returns
        -------
        data : :py:class:`~numpy.ndarray`
            ([len(key)], N_data) encoded samples.

            Follows the same indexing conventions as NumPy arrays.

            You will need an instance of
            :py:class:`~deepwave.tools.math.optim.Sampler` to manipulate the sample.
        """
        data = self._data[key]
        return data

    def encode(self):
        """
        Serialize data into buffer.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        raise NotImplementedError

    @classmethod
    def decode(cls, enc):
        """
        Decode dataset from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding, output of
            (:py:meth:`~deepwave.tools.math.optim.Dataset.encode`.

        Returns
        -------
        D : :py:class:`~deepwave.tools.math.optim.Dataset`
        """
        raise NotImplementedError

    def to_file(self, file_name):
        """
        Dump data to disk in .npz format.

        Parameters
        ----------
        file_name : str
            Name of file.
        """
        file_path = pathlib.Path(file_name).expanduser().absolute()
        np.savez(str(file_path), data=self.encode())

    @classmethod
    def from_file(cls, file_name):
        """
        Load dataset from .npz file.

        Parameters
        ----------
        file_name : str
            Name of file.

        Returns
        -------
        D : :py:class:`~deepwave.tools.math.optim.DataSet`
        """
        file_path = pathlib.Path(file_name).expanduser().absolute()
        data = np.load(str(file_path))['data']
        D = cls.decode(data)
        return D


class Sampler:
    """
    Serializer to encode/decode samples.
    """

    def __init__(self):
        pass

    def encode(self, buffer=None, **kwargs):
        """
        Encode sample information in buffer.

        Parameters
        ----------
        buffer : :py:class:`~numpy.ndarray`
            (N_cell,) buffer in which to write the data.
            If `None`, a new buffer will be allocated.
        kwargs : dict(:py:class:`~numpy.ndarray`)
            Fields to insert into buffer.
            If `None`, the buffer is not modified at the intended location.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        raise NotImplementedError

    def decode(self, enc):
        """
        Decode sample information from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            ([N_sample], N_cell) vectorized encoding, output of
            (:py:meth:`~deepwave.tools.math.optim.Sampler.encode`.

        Returns
        -------
        field_1 : :py:class:`~numpy.ndarray`
            ([N_sample], ...) first parameter.
        field_2 : :py:class:`~numpy.ndarray`
            ([N_sample], ...) second parameter.
        ...
        """
        raise NotImplementedError


class Parameter:
    """
    Serializer to encode/decode parameters.
    """

    def __init__(self):
        pass

    def encode(self, buffer=None, **kwargs):
        """
        Encode parameter information in buffer.

        Parameters
        ----------
        buffer : :py:class:`~numpy.ndarray`
            (N_cell,) buffer in which to write the data.
            If `None`, a new buffer will be allocated.
        kwargs : dict(:py:class:`~numpy.ndarray`)
            Fields to insert into buffer.
            If `None`, the buffer is not modified at the intended location.

        Returns
        -------
        enc : :py:class:`~numpy.ndarray`
            (N_cell,) vectorized encoding.
        """
        raise NotImplementedError

    def decode(self, enc):
        """
        Decode parameter information from buffer.

        Parameters
        ----------
        enc : :py:class:`~numpy.ndarray`
            ([N_sample], N_cell) vectorized encoding, output of
            (:py:meth:`~deepwave.tools.math.optim.Parameter.encode`.

        Returns
        -------
        field_1 : :py:class:`~numpy.ndarray`
            ([N_sample], ...) first parameter.
        field_2 : :py:class:`~numpy.ndarray`
            ([N_sample], ...) second parameter.
        ...
        """
        raise NotImplementedError


class ScalarFunction:
    """
    Proxy object to evaluate parameterized objectives and gradients.
    """

    def __init__(self):
        pass

    def eval(self, p, x):
        r"""
        Evaluate f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of some
            :py:meth:`~deepwave.tools.math.optim.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of some
            :py:meth:`~deepwave.tools.math.optim.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

        Returns
        -------
        z : float
            z = \frac{1}{N_sample} \sum_{i = 1}^{N_sample} f(p, x[i])
        """
        raise NotImplementedError

    def grad(self, p, x):
        r"""
        Evaluate \grad_{p} f(p, x).

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter encoding, output of some
            :py:meth:`~deepwave.tools.math.optim.Parameter.encode`.
        x : :py:class:`~numpy.ndarray`
            ([N_sample,], N_cell_2) vectorized sample encoding, output of some
            :py:meth:`~deepwave.tools.math.optim.Sampler.encode`.

            Several samples can be provided if stacked along axis 0.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            (N_cell_1,) vectorized parameter gradient, output of some
            :py:meth:`~deepwave.tools.math.optim.Parameter.encode`.

            z = \frac{1}{N_sample} \sum_{i=1}^{N_sample} \grad_{p}{f(p, x[i])}
        """
        raise NotImplementedError


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent Optimizer.

    Note
    ----
    This class assumes you have set up a logger from :py:namespace:`logger` beforehand.
    """

    def __init__(self, func, batch_size=32, N_epoch=10, alpha=1e-2, mu=0, verbosity='HIGH'):
        """
        Parameters
        ----------
        func : list(:py:class:`~deepwave.tools.math.optim.ScalarFunction`)
            One or more convex objective functions.
        batch_size : int
            Number of samples to process per batch.
        N_epoch : int
            Number of passes on the full dataset.
        alpha : float
            Gradient step size.
        mu : float
            Max momentum coefficient in [0, 1] for Nesterov acceleration from [1].
            Should be set to 0 or around 0.995
        verbosity : str
            ['LOW', 'HIGH']

        Notes
        -----
        [1] On the importance of initialization and momentum in deep learning
            [Sutskever, Martens, Dahl, Hinton]

        * `mu` is updated according to schedule given in [1].
        """
        self._func = func

        if batch_size < 1:
            raise ValueError('Parameter[batch_size] must be positive.')
        self._batch_size = int(batch_size)

        if N_epoch < 1:
            raise ValueError('Parameter[N_epoch] must be positive.')
        self._N_epoch = int(N_epoch)

        if alpha <= 0:
            raise ValueError('Parameter[alpha] must be positive.')
        self._alpha = alpha

        if not (0 <= mu <= 1):
            raise ValueError('Parameter[mu] must be in [0, 1].')
        self._mu = mu

        if not (verbosity in ['LOW', 'HIGH']):
            raise ValueError('Parameter[verbosity] must be {"LOW", "HIGH"}.')
        self._be_verbose = True if (verbosity == 'HIGH') else False

        # Buffers that hold optimization results
        self._p_opt = None
        self._iter_loss = None
        self._t_loss = None
        self._v_loss = None
        self._t = None

    def _epoch_print(self, epoch_idx, N_epoch, msg: str):
        msg = f'(E{epoch_idx:04d}/{N_epoch:04d}) {msg}'
        logging.info(msg)

    def _total_loss(self, p, D):
        """
        Compute loss function over entire dataset.

        Parameters
        ----------
        p : :py:class:`~numpy.ndarray`
            (N_cell,) parameter
        D : :py:class:`~deepwave.tools.math.optim.DataSet`

        Returns
        -------
        loss : float
        """
        N_sample = (len(D) // self._batch_size) * self._batch_size
        N_batch = N_sample // self._batch_size
        sample_idx = (np.random.permutation(np.arange(N_sample))
                      .reshape(N_batch, -1))

        loss = 0
        for i in range(N_batch):
            x = D[sample_idx[i]]
            for func in self._func:
                loss += func.eval(p, x)

        loss /= N_batch
        return loss

    def fit(self, ts, vs, p0, file_name=None):
        """
        Run optimization algorithm.

        After each epoch the outputs are written to disk.

        Parameters
        ----------
        ts : :py:class:`~deepwave.tools.math.optim.DataSet`
            Training set.
        vs : :py:class:`~deepwave.tools.math.optim.DataSet`
            Validation set.
        p0 : :py:class:`~numpy.ndarray`
            Initial point for optimization.
        file_name : str
            Name of .npz file to log output data to.

        Returns
        -------
        res : dict
            p_opt : :py:class:`~numpy.ndarray`
                (N_epoch + 1, N_cell) optimized parameter per epoch.
                `p_opt[0] = p0`
            iter_loss : :py:class:`~numpy.ndarray`
                (N_epoch, N_batch) loss function value per (epoch, batch) on
                training set.
            t_loss : :py:class:`~numpy.ndarray`
                (N_epoch + 1,) loss function value per epoch on training set.
            v_loss : :py:class:`~numpy.ndarray`
                (N_epoch + 1,) loss function value per epoch on validation set.
            t : :py:class:`~numpy.ndarray`
                (N_epoch,) execution time [s] per epoch.
                Includes time to compute training/validation loss.
        """
        N_sample = (len(ts) // self._batch_size) * self._batch_size
        N_batch = N_sample // self._batch_size

        N_cell = len(p0)
        p_opt = self._p_opt = np.zeros((self._N_epoch + 1, N_cell))
        iter_loss = self._iter_loss = np.zeros((self._N_epoch, N_batch))
        t_loss = self._t_loss = np.zeros((self._N_epoch + 1,))
        v_loss = self._v_loss = np.zeros((self._N_epoch + 1,))
        t = self._t = np.zeros((self._N_epoch,))

        p_opt[0] = p0
        self._epoch_print(0, self._N_epoch, 'Compute training loss')
        t_loss[0] = self._total_loss(p0, ts)
        self._epoch_print(0, self._N_epoch, 'Compute validation loss')
        v_loss[0] = self._total_loss(p0, vs)
        logging.info('')

        momentum = np.zeros_like(p0)
        momentum_idx = 0
        for epoch_idx in range(self._N_epoch):
            self._epoch_print(epoch_idx + 1, self._N_epoch, 'Permute samples')
            t_epoch_start = time.time()
            sample_idx = (np.random.permutation(np.arange(N_sample))
                          .reshape(N_batch, -1))
            p = p_opt[epoch_idx].copy()

            if not self._be_verbose:
                self._epoch_print(epoch_idx + 1, self._N_epoch, 'Perform batch-oriented steps')
            for batch_idx in range(N_batch):
                if self._be_verbose:
                    self._epoch_print(epoch_idx + 1, self._N_epoch, f'Batch {batch_idx + 1:04d}/{N_batch:04d}')
                x = ts[sample_idx[batch_idx]]

                momentum_idx += 1
                if np.isclose(self._mu, 0):
                    mu = 0
                else:
                    mu = 1 - (0.5 / (1 + np.floor(momentum_idx / self._batch_size)))
                    mu = min(mu, self._mu)

                grad_p = np.zeros_like(p)
                for func in self._func:
                    grad_p += func.grad(p + mu * momentum, x)
                momentum *= mu
                momentum -= self._alpha * grad_p
                p += momentum

                loss_p = 0
                for func in self._func:
                    loss_p += func.eval(p, x)
                iter_loss[epoch_idx, batch_idx] = loss_p

            p_opt[epoch_idx + 1] = p
            self._epoch_print(epoch_idx + 1, self._N_epoch, 'Compute training loss')
            t_loss[epoch_idx + 1] = self._total_loss(p, ts)
            self._epoch_print(epoch_idx + 1, self._N_epoch, 'Compute validation loss')
            v_loss[epoch_idx + 1] = self._total_loss(p, vs)
            t_epoch_end = time.time()
            t[epoch_idx] = t_epoch_end - t_epoch_start

            self._epoch_print(epoch_idx + 1, self._N_epoch, 'Save to disk')
            logging.info('')

            if file_name is not None:
                file_path = pathlib.Path(file_name).expanduser().absolute()
                np.savez(str(file_path),
                         p_opt=p_opt,
                         iter_loss=iter_loss,
                         t_loss=t_loss,
                         v_loss=v_loss,
                         t=t)

        logging.info('END')
        res = dict(p_opt=p_opt,
                   iter_loss=iter_loss,
                   t_loss=t_loss,
                   v_loss=v_loss,
                   t=t)
        return res
