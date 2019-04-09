import torch

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import os

from lstsq import lstsq_dense as lstsq
from optimizer import AdaptiveProximalGradient as APG
from utils import *


class NoScaler(torch.nn.Module):

    def __init__(self):
        super(NoScaler, self).__init__()

    def forward(self, X):
        return X

    def prox(self, t):
        pass


class Scaler(torch.nn.Module):

    def __init__(self, N, lambd=0):
        super(Scaler, self).__init__()
        self.omega_data = torch.nn.Parameter(
            torch.zeros(N), requires_grad=True)
        self.N = N
        self.lambd = lambd

    def forward(self, X):
        return torch.exp(self.omega_data)[:, None] * X

    def prox(self, t):
        # self.omega_data.data = torch.clamp(self.omega_data.data, -.5, .5)
        A_11 = (1. + self.lambd * t)
        A_12 = torch.ones(self.N, 1)
        A_21 = torch.ones(1, self.N)
        b_1 = self.omega_data.data
        S = (-A_21 @ (1. / A_11 * A_12)).item()
        b_tilde = -A_21 @ (1. / A_11 * b_1)
        x_2 = b_tilde / S
        x_1 = 1. / A_11 * (b_1 - A_12@x_2)
        self.omega_data.data = x_1


class NoRegularization(torch.nn.Module):

    def __init__(self, n):
        super(NoRegularization, self).__init__()
        self.n = n
        self.len = n

    def forward(self):
        return [2. * torch.eye(self.n)]

    def prox(self, t):
        pass


class Regularization(torch.nn.Module):

    def __init__(self, n):
        super(Regularization, self).__init__()
        self.omega_reg = torch.nn.Parameter(
            torch.tensor([0., 0.]), requires_grad=True)

        G = nx.grid_2d_graph(28, 28)
        U = nx.incidence_matrix(G, oriented=True)
        U = U.todense().T
        self.Rs = [torch.eye(n), torch.tensor(U).float()]
        self.len = sum([s.shape[0] for s in self.Rs])

    def forward(self):
        return [torch.exp(self.omega_reg[i]) * self.Rs[i] for i in range(len(self.Rs))]

    def prox(self, t):
        pass


class Regularization3(torch.nn.Module):

    def __init__(self):
        super(Regularization3, self).__init__()
        self.omega_reg = torch.nn.Parameter(
            1 * torch.ones(3), requires_grad=True)

        G = nx.grid_2d_graph(28, 28)
        U = nx.incidence_matrix(G, oriented=True).todense().T
        self.Rs = [torch.cat([torch.eye(784), torch.zeros(784, 100 + 1)], dim=1),
                   torch.cat(
                       [torch.zeros(100, 784), torch.eye(100), torch.zeros(100, 1)], dim=1),
                   torch.cat([torch.tensor(U).float(),
                              torch.zeros(1512, 100 + 1)], dim=1)
                   ]
        self.len = sum([s.shape[0] for s in self.Rs])

    def forward(self):
        return [torch.exp(self.omega_reg[i]) * self.Rs[i] for i in range(len(self.Rs))]

    def prox(self, t):
        pass


class NoFeatureEng(torch.nn.Module):

    def __init__(self):
        super(NoFeatureEng, self).__init__()

    def forward(self, x, mode='train'):
        return x

    def prox(self, t):
        pass


class FeatureEng(torch.nn.Module):

    def __init__(self, d_train, d_val, d_test):
        super(FeatureEng, self).__init__()
        self.d_train = d_train
        self.d_val = d_val
        self.d_test = d_test
        self.sigma = torch.nn.Parameter(0 * torch.ones(1), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, mode='train'):
        if mode == 'train':
            s = self.softmax(-self.d_train / torch.exp(self.sigma))
        if mode == 'val':
            s = self.softmax(-self.d_val / torch.exp(self.sigma))
        if mode == 'test':
            s = self.softmax(-self.d_test / torch.exp(self.sigma))
        return torch.cat([x, s, ones_like(x, (x.shape[0], 1))], dim=1)

    def prox(self, t):
        pass


def fit(U, Y, phi, data_scaler, reg_scaler):
    X = phi(U, "train")
    N, n = X.shape
    _, m = Y.shape
    A = torch.cat([data_scaler(X)] + reg_scaler(), dim=0)
    B = torch.cat([data_scaler(Y), zeros_like(Y, (reg_scaler.len, m))], dim=0)
    return lstsq(A, B)


def evaluate(x_train, y_train, x_val, y_val, x_test, y_test, phi, data, reg, optimize=True, **kwargs):
    # True loss
    loss = torch.nn.CrossEntropyLoss()
    psi = lambda theta, U, Y, mode: loss(phi(U, mode).detach()@theta, Y.argmax(dim=1))

    def eval_loss():
        theta = fit(x_train, y_train, phi, data, reg)
        return psi(theta, x_val, y_val, mode="val")

    def prox(t):
        data.prox(t)
        reg.prox(t)
        phi.prox(t)

    def test_error():
        theta = fit(x_train, y_train, phi, data, reg)
        val_loss = psi(theta, x_val, y_val, mode="val").item()
        test_loss = psi(theta, x_test, y_test, mode="test").item()
        val_error = (y_val.argmax(dim=1) != (phi(x_val, 'val')@theta).argmax(dim=1)).float().mean().item()
        test_error = (y_test.argmax(dim=1) != (phi(x_test, 'test')@theta).argmax(dim=1)).float().mean().item()
        print("val_loss: %.2f" % val_loss)
        print("test_loss: %.2f" % test_loss)
        print("val_error: %.1f" % (val_error * 100))
        print("test_error: %.1f" % (test_error * 100))

    optim = APG([
        {"params": data.parameters()},
        {"params": reg.parameters(), "lr": 1.},
        {"params": phi.parameters(), "lr": 1.}
    ], lr=1., prox=prox)

    if optimize:
        optim.run(eval_loss, **kwargs)
    test_error()


def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(12)

    torch.manual_seed(0)
    np.random.seed(0)

    for size in [50_000, 5_000]:
        # Pre-processing
        x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np = get_mnist(
            size)
        x_train_np = np.array([deskew(x_train_np[i].reshape(
            28, 28), (28, 28), negated=True).flatten()for i in range(x_train_np.shape[0])])
        x_val_np = np.array([deskew(x_val_np[i].reshape(
            28, 28), (28, 28), negated=True).flatten() for i in range(x_val_np.shape[0])])
        x_test_np = np.array([deskew(x_test_np[i].reshape(
            28, 28), (28, 28), negated=True).flatten() for i in range(x_test_np.shape[0])])
        x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np = map(lambda x: x.astype(
            float), [x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np])
        _, m = y_train_np.shape
        N, n = x_train_np.shape
        Nval = x_val_np.shape[0]
        Ntest = x_test_np.shape[0]
        print("\nN =", N, "Nval =", Nval, "Ntest =",
              Ntest, "n =", n, "m =", str(m) + ":")

        # Scaling
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_np)
        x_val = scaler.transform(x_val_np)
        x_test = scaler.transform(x_test_np)
        x_train, y_train, x_val, y_val, x_test, y_test = map(
            lambda x: torch.tensor(x).float(),
            [x_train, y_train_np, x_val, y_val_np, x_test, y_test_np]
        )

        print("LS")
        phi = NoFeatureEng()
        data = NoScaler()
        reg = NoRegularization(784)
        evaluate(x_train, y_train, x_val, y_val, x_test,
                 y_test, phi, data, reg, optimize=False)

        print("\nLS + reg x 2")
        phi = NoFeatureEng()
        data = NoScaler()
        reg = Regularization(784)
        evaluate(x_train, y_train, x_val, y_val, x_test,
                 y_test, phi, data, reg, eps=1e-4, maxiter=250)

        # Feature engineering
        x_train, y_train, x_val, y_val, x_test, y_test = map(
            lambda x: torch.tensor(x).float(),
            [x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np]
        )
        archetypes = torch.tensor(get_features(
            x_train_np, y_train_np)).float().cpu()
        distances = (x_train.cpu().unsqueeze(-1) - archetypes.unsqueeze(0))
        d_train = torch.norm(distances, dim=1).cuda()
        distances = (x_val.cpu().unsqueeze(-1) - archetypes.unsqueeze(0))
        d_val = torch.norm(distances, dim=1).cuda()
        distances = (x_test.cpu().unsqueeze(-1) - archetypes.unsqueeze(0))
        d_test = torch.norm(distances, dim=1).cuda()

        print("\nLS + reg x 3 + feat")
        phi = FeatureEng(d_train, d_val, d_test)
        data = NoScaler()
        reg = Regularization3()
        evaluate(x_train, y_train, x_val, y_val, x_test,
                 y_test, phi, data, reg, eps=1e-4, maxiter=250)

        print("\nLS + reg x 3 + feat + data weighting")
        phi = FeatureEng(d_train, d_val, d_test)
        data = Scaler(N, 1e-3)
        reg = Regularization3()
        evaluate(x_train, y_train, x_val, y_val, x_test,
                 y_test, phi, data, reg, eps=1e-4, maxiter=250)
        print(torch.min(data.omega_data), torch.max(data.omega_data))

        if size == 5_000:
            os.makedirs("figs", exist_ok=True)
            for i, j in enumerate(data.omega_data.argsort()[:6]):
                print("outlier", i, np.argmax(y_train_np[j]))
                plt.imshow(x_train_np[j].reshape(28, 28), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('figs/outlier_%d.pdf' % i)
                plt.close()

            for i, j in enumerate(data.omega_data.argsort()[-6:]):
                print("exemplar", i, np.argmax(y_train_np[j]))
                plt.imshow(x_train_np[j].reshape(28, 28), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('figs/exemplar_%d.pdf' % i)
                plt.close()

if __name__ == '__main__':
    main()
