import torch
import copy


class AdaptiveProximalGradient(torch.optim.Optimizer):

    def __init__(self, params, lr=1., increase_rate=1.2, decrease_rate=2., prox=None):
        defaults = dict(lr=lr, increase_rate=increase_rate,
                        decrease_rate=decrease_rate, prox=prox)
        super(AdaptiveProximalGradient, self).__init__(params, defaults)

    def increase_step_size(self):
        for group in self.param_groups:
            group['lr'] *= group['increase_rate']

    def decrease_step_size(self):
        for group in self.param_groups:
            group['lr'] /= group['decrease_rate']

    def save_state(self):
        param_state = []
        grad_state = []
        for group in self.param_groups:
            param_state += [[copy.deepcopy(p.data) for p in group['params']]]
            grad_state += [[copy.deepcopy(p.grad.data)
                            for p in group['params']]]
        return (param_state, grad_state)

    def restore_state(self, state):
        param_state, grad_state = state
        for i, group in enumerate(self.param_groups):
            for p, val in zip(group['params'], param_state[i]):
                p.data = val
            for p, g in zip(group['params'], grad_state[i]):
                if g is None:
                    continue
                p.grad = g

    def stopping_criterion(self, state):
        param_state, grad_state = state
        stopping_crit_squared = 0.
        for i, group in enumerate(self.param_groups):
            for p, val, g in zip(group['params'], param_state[i], grad_state[i]):
                if g is None:
                    continue
                stopping_crit_squared += torch.sum(
                    ((val - p.data) / group['lr'] + (p.grad.data - g))**2)
        return torch.sqrt(stopping_crit_squared).item()

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-group['lr'], p.grad.data)

    def prox(self):
        for group in self.param_groups:
            group['prox'](group['lr'])

    def run(self, eval_loss, eps=.1, maxiter=10000, callback=None, verbose=False):
        """Runs the optimization procedure until convergence.

        Arguments:
            loss (callable): A closure that reevaluates the model
                and returns the loss.
            prox (callable): A method that performs a proximal step
                on all the parameters.
            eps (float, optional): stopping criterion.
            maxiter (float, optional): maximum number of iterations.
        """

        k = 0
        loss = eval_loss()
        loss.backward()
        while True:
            state = self.save_state()
            self.step()
            self.prox()
            new_loss = eval_loss()
            if new_loss.item() <= loss.item():
                loss = new_loss
                self.zero_grad()
                loss.backward()
                stop = self.stopping_criterion(state)
                if verbose:
                    print(stop, loss.item(), self.param_groups[0]['lr'])
                if callback is not None:
                    callback()
                if stop < eps:
                    break
                self.increase_step_size()
            else:
                self.restore_state(state)
                self.decrease_step_size()
            k += 1
            if k == maxiter:
                break
        return loss.item()
