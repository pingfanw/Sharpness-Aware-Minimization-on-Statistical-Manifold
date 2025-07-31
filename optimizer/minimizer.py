import torch
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.1, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            t_w = self.state[param].get("eps")
            if t_w is None:
                t_w = torch.clone(param).detach()
                self.state[param]["eps"] = t_w
            if 'weight' in name:
                t_w[...] = param[...]
                t_w.abs_().add_(self.eta)
                param.grad.mul_(t_w) 
            wgrads.append(torch.norm(param.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2).item() + 1.e-16
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            t_w = self.state[param].get("eps")
            if 'weight' in name:
                param.grad.mul_(t_w)
            eps = t_w   
            eps[...] = param.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            param.data.add_(eps)
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grads.append(torch.norm(param.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2).item() + 1.e-16     
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            eps = self.state[param].get("eps")
            if eps is None:
                eps = torch.clone(param).detach()
                self.state[param]["eps"] = eps
            eps[...] = param.grad[...]
            eps.mul_(self.rho / grad_norm)   
            param.data.add_(eps)       
        self.optimizer.zero_grad()

class SMSAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        fvp_eps = []
        fvp_list = []
        gradT_list = []
        all_bias = True
        for m in self.optimizer.modules:
            if self.optimizer.steps % self.optimizer.TInv ==0:
                self.optimizer._update_inv(m)
            param_grad_mat = self.optimizer._get_matrix_form_grad(m, m.__class__.__name__)
            gradT_list.append(param_grad_mat.view(1,-1))
            fvp_buffer = self.optimizer._get_natural_grad(m, param_grad_mat)
            fvp_eps.append(fvp_buffer[0])
            fvp_buffer[0] = fvp_buffer[0].view(m.weight.grad.data.size(0), -1)
            if m.bias is not None:
                fvp_eps.append(fvp_buffer[1])
                fvp_buffer[1] = fvp_buffer[1].view(m.bias.grad.data.size(0), 1)
                fvp_list.append(torch.cat(fvp_buffer,dim=1).view(-1, 1))
            else:
                fvp_list.append(fvp_buffer[0].view(-1, 1))
                all_bias = False
        num_modules = len(gradT_list)
        if all_bias == True:
            num_modules *= 2
        # row vectorized grad
        gradT = torch.cat(gradT_list,dim=1)
        # column vectorized fvp
        fvp = torch.cat(fvp_list,dim=0)
        # computes norm
        fvp_norm = torch.sqrt(gradT @ fvp).item() + 1.e-16
        index = 0
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None and index >= num_modules:
                    continue
                eps = self.state[param].get("eps")
                if eps is None:
                    eps = torch.clone(param).detach()
                    self.state[param]["eps"] = eps
                eps[...] = fvp_eps[index][...]
                eps.mul_(self.rho/fvp_norm)   
                param.data.add_(eps)
                index += 1
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for m in self.optimizer.modules:
            for param in m.parameters():
                if param.grad is None:
                    continue
                param.data.sub_(self.state[param]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()