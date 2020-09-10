import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import utils


# define the GNBNTM neural network
class GNBNTM(nn.Module):
    def __init__(self, device, vocab_num, hidden_num, topic_num, shape_prior, scale_prior):
        super(GNBNTM, self).__init__()
        self.device = device
        self.shape_prior = shape_prior
        self.scale_prior = scale_prior

        self.hidden_fc = nn.Linear(vocab_num, hidden_num)
        self.gamma_fc = nn.Linear(hidden_num, topic_num)
        self.scale_fc = nn.Linear(hidden_num, topic_num)

        self.pj_fc = nn.Linear(hidden_num, 1)
        self.out_fc = nn.Linear(topic_num, vocab_num)

    def forward(self, batch):
        doc = torch.tanh(self.hidden_fc(batch))
        gamma = F.softplus(self.gamma_fc(doc))
        scale = F.softplus(self.scale_fc(doc))
        alpha = self.sample_gamma(gamma, scale)
        pj = F.softplus(self.pj_fc(doc))

        lam = self.sample_gamma(alpha, pj)
        n = self.sample_poisson(lam)
        out = self.out_fc(torch.sigmoid(n + alpha))

        return gamma, scale, lam, out

    def sample_gamma(self, shape, scale):
        augment = 10
        # get Gamma(shape + factor, 1)
        with torch.no_grad():
            sample = distributions.Gamma(shape + augment, 1).sample()
            eps = torch.sqrt(9. * (shape + augment) - 3.) * (
                    ((sample / (shape + augment - (1. / 3.))) ** (1. / 3.)) - 1.)

        z = (shape + augment - (1. / 3.)) * ((1. + (eps / torch.sqrt(9. * (shape + augment) - 3.))) ** 3.)

        # reduce factor
        with torch.no_grad():
            expand_shape = shape.unsqueeze(-1).repeat(1, 1, augment)
            factor_range = torch.arange(1, augment + 1, dtype=torch.float, device=self.device).expand_as(expand_shape)
            u = distributions.Uniform(torch.zeros(factor_range.size(), device=self.device),
                                      torch.ones(factor_range.size(), device=self.device)).sample()

        u_prod = torch.prod(u ** (1. / (expand_shape + factor_range - 1. + 1e-12)), -1)
        z = z * u_prod * scale

        return z

    def sample_poisson(self, lam):
        # get Poisson(lam)
        with torch.no_grad():
            sample = distributions.Poisson(lam).sample()
            eps = (sample - lam) / torch.sqrt(lam + 1e-12)

        z = torch.sqrt(lam + 1e-12) * eps + lam

        return z

    def compute_batch_loss(self, x, y, shape, scale):
        # compute likelihood
        likelihood = -torch.sum(torch.log_softmax(y, 1) * x, 1)

        # compute KL divergence
        prior_distribution = distributions.Gamma(self.shape_prior, self.scale_prior)
        local_distribution = distributions.Gamma(shape, scale)
        kld = torch.sum(distributions.kl_divergence(local_distribution, prior_distribution), 1)

        return likelihood, kld

