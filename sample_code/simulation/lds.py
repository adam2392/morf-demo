import matplotlib.pyplot as plt
import numpy as np
import mne
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


class DiscreteLinearSystem:
    """An abstraction of a discrete time linear system."""

    def __init__(self, A, B=None):
        self.A = A
        self.B = B

        # compute eigenvalues
        self.eigs = np.linalg.eigvals(self.A)

    @property
    def cond(self):
        """Return the condition number of the state matrix."""
        return np.linalg.cond(self.A)

    def reconstruct(self, x0, ut=None, steps=100, add_noise=True):
        """Reconstruct the data given initial condition."""
        if len(x0) != self.A.shape[0]:
            raise ValueError("x0 needs to be the same dimension as state matrix, A.")

        if ut is not None:
            if self.B is not None and ut.shape[0] != self.B.shape[1]:
                raise ValueError("ut needs to be the same dimension as state matrix, B.")

            if ut.shape[1] != steps:
                raise RuntimeError("Input should be the same length as the desired number of steps")

        # store the results as list
        xhat = [x0]

        # simulate forward in time
        for i in range(steps - 1):
            lti_response = self.A.dot(xhat[-1])
            input_response = self.B.dot(ut[:, i]) if ut is not None else 0
            if add_noise:
                y = np.random.normal(size=(len(x0),))
            else:
                y = np.zeros((len(x0),))

            # print(lti_response.shape, input_response.shape, y.shape)
            xhat.append(lti_response + input_response + y)

        return np.array(xhat).T

    def mse(self, xtrue, xhat):
        """Compute Mean-Squared Error."""
        # compute mse
        mse = np.mean((xhat.flatten() - xtrue.flatten()) ** 2)
        return mse

    def compute_gersh_discs(self, A, picks=None):
        """Compute the G-discs for matrix A."""
        if picks is None:
            picks = np.arange(len(A))

        # store radii and 2D centroids as lists
        radii = []
        centroids = []

        # loop over matrix and compute G-radii
        for i in picks:
            xi = np.real(A[i, i])
            yi = np.imag(A[i, i])
            ri = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            radii.append(ri)
            centroids.append((xi, yi))

        return radii, centroids

    def plot_gerschgorin_discs(self, A, picks=None, ax=None):
        """Plot G-discs for a matrix A."""
        n = len(A)
        eval, evec = np.linalg.eig(A)

        patches = []

        # draw discs
        radii, centroids = self.compute_gersh_discs(A, picks=picks)
        for i in range(len(radii)):
            xi, yi = centroids[i]
            ri = radii[i]
            circle = Circle((xi, yi), ri)
            patches.append(circle)

        if ax is None:
            fig, ax = plt.subplots()

        p = PatchCollection(patches, cmap="turbo", alpha=0.1)
        ax.add_collection(p)
        plt.axis("equal")

        for xi, yi in zip(np.real(eval), np.imag(eval)):
            plt.plot(xi, yi, "o")
        plt.show()
        return fig, ax, radii

    def plot_corr_reconstruction(self, xtrue, train_size, ax=None):
        """Plot values correlated against each other."""
        if xtrue.ndim != 2:
            raise ValueError("xtrue passed in should be a 2D array " "of size C x T.")

        test_size = xtrue.shape[1] - train_size
        # get initial conditions
        x0_train = xtrue[:, 0]
        x0_test = xtrue[:, train_size]

        # splice training and testing data
        xtrain = xtrue[:, :train_size]
        xtest = xtrue[:, train_size:]

        # perform reconstruction using linear model
        xtrain_hat = self.reconstruct(x0_train, steps=train_size - 1)
        xtest_hat = self.reconstruct(x0_test, steps=test_size - 1)

        # plot performance
        if ax is None:
            fig, ax = plt.subplots()

        train_mse = self.mse(xtrain, xtrain_hat)
        test_mse = self.mse(xtest, xtest_hat)
        train_points = ax.plot(
            xtrain.flatten(),
            xtrain_hat.flatten(),
            "bo",
            label="train-err={}".format(train_mse),
        )
        test_points = ax.plot(
            xtest.flatten(),
            xtest_hat.flatten(),
            "ko",
            label="train-err={}".format(test_mse),
        )

        # for ich in range(xtrain.shape[0]):
        #     train_points = ax.plot(xtrain[ich,:], xtrain_hat[ich,:], 'bo', label="train-err={}".format(train_mse))
        #     test_points = ax.plot(xtest[ich,:], xtest_hat[ich,:], 'ko', label='train-err={}'.format(test_mse))
        # ax.plot([0, train_size], [0, 1], 'r', label='perfect-correlation')
        ax.set_xlabel("True data points")
        ax.set_ylabel("Reconstructed data points")
        ax.legend(
            # [train_points, test_points]
        )

    def plot_psd(self, xtrue, winsize, info, ax=None):
        """Plot power spectrum density."""
        # get initial conditions
        x0 = xtrue[:, 0]

        # perform reconstruction using linear model
        x_hat = self.reconstruct(x0, steps=winsize - 1)

        # create mne raw arrays to make plotting psd easy
        raw_true = mne.io.RawArray(xtrue, info)
        raw_hat = mne.io.RawArray(x_hat, info)

        # plot psds
        if ax is None:
            fig, ax = plt.subplots()

        raw_true.plot_psd(
            # fmin=fmin, fmax=fmax,
            ax=ax,
            color="blue",
            average=True,
            show=False,
        )

        raw_hat.plot_psd(
            # fmin=fmin, fmax=fmax,
            ax=ax,
            color="red",
            average=True,
            show=False,
        )
        return ax

    def plot_eigs(self, ax=None, show_unit_circle=True, show_axes=True):
        """Plot eigenvalues with unit circle."""
        if ax is None:
            fig, ax = plt.subplots()

        points = ax.plot(
            np.real(self.eigs), np.imag(self.eigs), "bo", label="Eigenvalues"
        )

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(self.eigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        ax.set_ylabel("Imaginary part")
        ax.set_xlabel("Real part")

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0.0, 0.0),
                1.0,
                color="green",
                fill=False,
                label="Unit circle",
                linestyle="--",
            )
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("-.")
        ax.grid(True)
        ax.set_aspect("equal")

        # x and y axes
        if show_axes:
            ax.annotate(
                "",
                xy=(np.max([limit * 0.8, 1.0]), 0.0),
                xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                arrowprops=dict(arrowstyle="->"),
            )
            ax.annotate(
                "",
                xy=(0.0, np.max([limit * 0.8, 1.0])),
                xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                arrowprops=dict(arrowstyle="->"),
            )

        # legend
        if show_unit_circle:
            ax.add_artist(
                plt.legend([points, unit_circle], ["Eigenvalues", "Unit circle"], loc=1)
            )
        else:
            ax.add_artist(plt.legend([points], ["Eigenvalues"], loc=1))

        return ax
