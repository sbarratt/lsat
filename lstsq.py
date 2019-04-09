import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, gmres
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator


class DenseLeastSquares(torch.autograd.Function):
    r"""Custom autograd function that solves the matrix least
    squares problem: minimize ||A * theta - B||_F^2
    """

    @staticmethod
    def forward(ctx, A, B):
        r"""In the forward pass we solve the matrix least-squares
        problem given the problem data A and B. We cache A, B, theta,
        and u (the cholesky decomposition).

        Args:
            A: p x n Tensor.
            B: p x m Tensor.

        Returns:
            theta^ls: Solution to matrix least-squares problem.
        """
        # compute gram matrix of A, and its QR factorization
        with torch.no_grad():
            u = torch.cholesky(A.t() @ A, upper=True)
            theta = torch.potrs(A.t() @ B, u)

        # cache information needed for backward pass
        ctx.save_for_backward(A, B, theta, u)

        return theta

    @staticmethod
    def backward(ctx, dtheta):
        r"""In the backward pass we receive a Tensor containing
        the gradient of the loss with respect to theta, and we compute
        the gradient of the loss with respect to the input (A and B)
        using the chain rule.

        Args:
            dtheta: n x m Tensor.

        Returns:
            dA: Gradient of loss wrt A, a p x n Tensor.
            dB: Gradient of loss wrt B, a p x m Tensor.
        """
        A, B, theta, u = ctx.saved_tensors

        with torch.no_grad():
            C = torch.potrs(dtheta, u)
            dA = (B - A@theta)@C.t() - A@C@theta.t()
            dB = A @ C

        return dA, dB


def sparse_numpy_to_torch(A, cuda=False):
    Acoo = A.tocoo()
    i = torch.LongTensor(np.vstack((Acoo.row, Acoo.col)))
    v = torch.DoubleTensor(Acoo.data)
    shape = A.shape

    if cuda:
        return torch.cuda.sparse.DoubleTensor(i, v, torch.Size(shape))
    else:
        return torch.sparse.DoubleTensor(i, v, torch.Size(shape))


def sparse_torch_to_numpy(A):
    data = A._values().cpu().data.numpy()
    row_ind = A._indices()[0, :].cpu().data.numpy()
    col_ind = A._indices()[1, :].cpu().data.numpy()
    shape = A.shape
    return sparse.csc_matrix((data, (row_ind, col_ind)), (shape[0], shape[1]))


def compose_linear_ops(A, B):
    def matvec(x):
        return A.matvec(B.matvec(x))

    def rmatvec(y):
        return B.rmatvec(A.rmatvec(y))

    return LinearOperator((A.shape[0], B.shape[1]), matvec=matvec, rmatvec=rmatvec)


class SparseLeastSquares(torch.autograd.Function):
    r"""Custom autograd function that solves the matrix least
    squares problem: minimize ||A * theta - B||_F^2, where A is sparse.
    """

    @staticmethod
    def forward(ctx, A, B):
        r"""In the forward pass we solve the matrix least-squares
        problem given the problem data A and B.

        Args:
            A: p x n sparse.FloatTensor.
            B: p x m Tensor.

        Returns:
            theta^ls: Solution to matrix least-squares problem.
        """
        Anp = sparse_torch_to_numpy(A)
        Bnp = B.cpu().data.numpy()
        X = []
        for i in range(Bnp.shape[1]):
            bnp = Bnp[:, i]
            X += [lsqr(Anp, bnp)[0]]
        thetanp = np.array(X).T

        ctx.Anp = Anp
        ctx.Bnp = Bnp
        ctx.thetanp = thetanp
        ctx.A = A

        return torch.Tensor(thetanp)

    @staticmethod
    def backward(ctx, dtheta):
        r"""In the backward pass we receive a Tensor containing
        the gradient of the loss with respect to theta, and we compute
        the gradient of the loss with respect to the input (A and B)
        using the chain rule.

        Args:
            dtheta: n x m Tensor.

        Returns:
            dA: Gradient of loss wrt A, a p x n Tensor.
            dB: Gradient of loss wrt B, a p x m Tensor.
        """

        dthetanp = dtheta.cpu().data.numpy()
        X = []
        AtAnp = compose_linear_ops(aslinearoperator(
            ctx.Anp.T), aslinearoperator(ctx.Anp))
        for i in range(dthetanp.shape[1]):
            bnp = dthetanp[:, i]
            X += [gmres(AtAnp, bnp)[0]]

        C = np.array(X).T

        Cthetat = C @ ctx.thetanp.T
        S = Cthetat + Cthetat.T
        rows = ctx.A._indices()[0, :].cpu().data.numpy()
        cols = ctx.A._indices()[1, :].cpu().data.numpy()
        vals = (ctx.Bnp[rows, :] * C.T[:, cols].T).sum(axis=1) - np.asarray(
            np.multiply(ctx.Anp[rows, :].todense(), S[:, cols].T)).sum(axis=1)
        dA = sparse.coo_matrix((vals, (rows, cols)), shape=ctx.Anp.shape)
        dB = ctx.Anp @ C

        return sparse_numpy_to_torch(dA), torch.Tensor(dB)

lstsq_dense = DenseLeastSquares.apply
lstsq_sparse = SparseLeastSquares.apply

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    k = 30
    n = 20
    m = 3
    Anp = np.random.randn(k, n)
    Asp = sparse_numpy_to_torch(sparse.csc_matrix(Anp)).requires_grad_(True)
    B = torch.DoubleTensor(k, m).normal_().requires_grad_(True)
    theta = lstsq_sparse(Asp, B)
    theta.sum().backward()

    torch.set_default_dtype(torch.float64)
    A = torch.from_numpy(Anp).requires_grad_(True)
    theta = lstsq_dense(A, B)
    theta.sum().backward()
    assert torch.mean(torch.abs(A.grad - Asp.grad)) < 1e-4

    # perform gradcheck
    p = 30
    n = 20
    m = 3
    for _ in range(5):
        A = torch.DoubleTensor(p, n).normal_().requires_grad_(True)
        B = torch.DoubleTensor(p, m).normal_().requires_grad_(True)
        assert torch.autograd.gradcheck(lstsq_dense, (A, B))

    print("All tests passed!")
    # generate random data
    # for k, n, m in [[20000, 10000, 10000], [20000, 10000, 1], [100000, 1000, 100]]:
    #     print("===========================================")
    #     print('k=%d, n=%d, m=%d' % (k, n, m))
    #     A = torch.cuda.DoubleTensor(k, n).normal_()
    #     B = torch.cuda.DoubleTensor(k, m).normal_()
    #     omega = torch.cuda.DoubleTensor(k).normal_().requires_grad_(True)

    #     with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #         At = A * torch.exp(omega / 2).view(k, 1)
    #         Bt = B * torch.exp(omega / 2).view(k, 1)
    #         theta = lstsq(At, Bt)
    #         psi = theta.sum()
    #     print(prof.table(sort_by='cuda_time_total'))
    #     with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #         psi.backward()
    #     print(prof.table(sort_by='cuda_time_total'))
    #     print("\n\n\n")
