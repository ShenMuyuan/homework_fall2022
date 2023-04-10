import abc
import itertools
from typing import Any
from torch import nn, Tensor
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = None
            self.optimizer = optim.Adam(self.mean_net.parameters(),
                                        self.learning_rate)
            # remove logstd because it is not learned
            # self.logstd = nn.Parameter(
            #     torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            # )
            # self.logstd.to(ptu.device)
            # self.optimizer = optim.Adam(
            #     itertools.chain([self.logstd], self.mean_net.parameters()),
            #     self.learning_rate
            # )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim > 1:
            # array of multiple observations
            # do nothing
            observation = obs
        else:
            # array of single observation
            # add a new dimension of size 1 to the first dimension
            observation = obs[np.newaxis, :]
            # observation = obs[None, :]  # OK
            # observation = obs[None]  # OK

        # convert ndarray to tensor
        observation_tensor = ptu.from_numpy(observation)
        # get action distribution from nn
        action_result = self.forward(observation_tensor)
        # not sampling from distribution, because standard deviation is not learned (always one)
        # print("action_result: ", action_result)
        # action_distribution = distributions.Categorical(logits=action_result) if self.discrete \
        #     else distributions.Normal(action_result, self.logstd.exp().expand_as(action_result))
        # print("mean: ", action_distribution.mean)
        # print("stddev: ", action_distribution.stddev)
        # sampled_action: torch.Tensor = action_distribution.sample()
        # print("sampled_action: ", sampled_action)
        return ptu.to_numpy(action_result)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # overriden
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.discrete:
            return self.logits_na(observation)
        else:
            return self.mean_net(observation)


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations: np.ndarray, actions: np.ndarray,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        self.optimizer.zero_grad()
        obs = ptu.from_numpy(observations)
        # actions of experts
        actions_expert = ptu.from_numpy(actions)
        # get action (mean) from nn
        actions_current = self.forward(obs)
        # calculate loss
        loss = self.loss(actions_current, actions_expert)
        # backpropagation
        loss.backward()
        # optimize
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
