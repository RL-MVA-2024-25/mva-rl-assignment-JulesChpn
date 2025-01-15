"""Training functions"""

import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import typing

from copy import deepcopy
from typing import Any, Dict, List

from env_hiv import HIVPatient
import names
import config


### UTILIY FUNCTIONS


def greedy_action(network: nn.Module, state: List[float]) -> int:
    """
    Choose the greedy action.

    Args:
        params (Dict[str, Any]): Parameters of the agent.
        network (nn.Module): Neural network.
        state (List[float]): State at previous time.

    Returns:
        int: Greedy action to take.
    """
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to("cpu"))
        return torch.argmax(Q).item()


def create_criterion(loss: str):
    """
    Create loss function.

    Args:
        loss (str): Name of the loss function.

    Raises:
        ValueError: If name of loss function is unknown.

    Returns:
        Loss function.
    """
    if loss == names.SMOOTH_L1:
        return nn.SmoothL1Loss()
    else:
        raise ValueError("This loss function is not supported")


def create_optimizer(optimizer: str, network: nn.Module, lr: float):
    """
    Create optimizer.

    Args:
        optimizer (str): Optimizer name.
        network (nn.Module): Neural network.
        lr (float): Learning rate.

    Raises:
        ValueError: If the optimizer name is unknown.

    Returns:
        Optimizer.
    """
    if optimizer == names.ADAM:
        return torch.optim.Adam(network.parameters(), lr=lr)
    else:
        raise ValueError("This optimizer is not supported")


_Memory = typing.TypeVar(name="_Memory", bound="Memory")


class Memory:
    def __init__(self: _Memory) -> None:
        """
        Initialize class instance.

        Args:
            self (_Memory): Class instance.
        """
        self.max_memory = 40000
        self.curr_memory = []
        self.position = 0
        self.device = "cpu"

    def append(
        self: _Memory,
        state: List[float],
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Add last state, action, reward... to memory.

        Args:
            self (_Memory): Class instance.
            state (List[float]): Previous state.
            action (int): Action.
            reward (float): Reward.
            next_state (int): Next state.
            done (bool): Whether the simulation is over or not.
        """
        if len(self.curr_memory) < self.max_memory:
            self.curr_memory.append(None)
        self.curr_memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_memory

    def sample(self: _Memory, batch_size: int) -> List:
        """
        Sample elements from the memory.

        Args:
            self (_Memory): Class instance.
            batch_size (int): Number of elements to sample.

        Returns:
            List: List of samples.
        """
        batch = random.sample(self.curr_memory, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self: _Memory) -> int:
        return len(self.curr_memory)


_DQN = typing.TypeVar(name="_DQN", bound="DQN")


class DQN:
    def __init__(self: _DQN) -> None:
        """
        Initialize class instance.

        Args:
            self (_DQN): Class instance.
            params (Dict[str, Any]): Parameters of teh agent.
        """
        self.memory = Memory()
        self.network = self.create_network().to("cpu")

        self.best_reward = -float("inf")
        self.epoch_rewards = []

    def create_network(self: _DQN) -> nn.Module:
        """
        Create the network of the DQN.

        Args:
            self (_DQN): CLass instance.

        Returns:
            nn.Module: Neural network.
        """
        layers = [
            nn.Linear(6, 256),
            nn.ReLU(),
        ]
        for _ in range(2):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                256,
                4,
            )
        )
        network = nn.Sequential(*layers)
        return network

    def gradient_step(self: _DQN) -> None:
        """
        Make a gradient step.

        Args:
            self (_DQN): Class instance.
        """
        if len(self.memory) > 1000:
            X, A, R, Y, D = self.memory.sample(1000)
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=0.90)
            QXA = self.network(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self: _DQN, env: HIVPatient):
        """
        Train the model.

        Args:
            self (_DQN): Class instance.
            env (HIVPatient): Environment.
        """
        network = self.create_network()
        target_network = deepcopy(network).to("cpu")
        best_model = self.network
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
        epoch = 0
        epoch_cum_reward = 0
        state, _ = env.reset()
        epsilon = 1.0
        step = 0
        max_reward = -float("inf")
        epsilon_step = (1.0 - 0.07) / 40000
        while epoch < 500:
            start_time = time.time()
            if step > 500:
                epsilon = max(
                    0.07,
                    epsilon - epsilon_step,
                )
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(network=network, state=state)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            epoch_cum_reward += reward
            for _ in range(2):
                if len(self.memory) > 1000:
                    X, A, R, Y, D = self.memory.sample(1000)
                    QYmax = target_network(Y).max(1)[0].detach()
                    update = torch.addcmul(R, 1 - D, QYmax, value=0.90)
                    QXA = network(X).gather(1, A.to(torch.long).unsqueeze(1))
                    loss = criterion(QXA, update.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            target_state_dict = target_network.state_dict()
            network_state_dict = network.state_dict()
            tau = 0.005
            for key in network_state_dict:
                target_state_dict[key] = (
                    tau * network_state_dict[key] + (1 - tau) * target_state_dict[key]
                )
            target_network.load_state_dict(target_state_dict)
            step += 1
            if done or trunc:
                end_time = time.time()
                print(f"Epoch {epoch+1} --------------------------------------")
                print(
                    f"Reward : {epoch_cum_reward/1e10:.2f}.1e10 --- Time : {end_time-start_time:.2f} seconds"
                )
                self.epoch_rewards.append(epoch_cum_reward)
                if epoch_cum_reward > max_reward:
                    best_model = network
                    best_reward = float(epoch_cum_reward)
                    max_reward = epoch_cum_reward
                epoch += 1
                state, _ = env.reset()
                epoch_cum_reward = 0
            else:
                state = next_state
        print("Training done.")
        print(f"Best reward : {best_reward/1e10:.2f}.1e10")
        return best_model


_ProjectAgent = typing.TypeVar(name="_ProjectAgent", bound="ProjectAgent")


class ProjectAgent:
    def __init__(self: _ProjectAgent) -> None:
        """
        Initialize class instance.

        Args:
            self (_ProjectAgent): Class instance.
        """
        self.id_experiment = 8
        self.model = DQN()
        self.best_model = self.model.create_network()

    def act(
        self: _ProjectAgent, observation: List[float], use_random: bool = False
    ) -> int:
        """
        Decide whith action to take given an observation.

        Args:
            self (_ProjectAgent): Class instance.
            observation (List[float]): Observation.
            use_random (bool, optional): Whether to do a random aciton or not. Defaults to False.

        Returns:
            int: Action to take.
        """
        if use_random:
            return 0
        self.best_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to("cpu")
            Q_values = self.best_model(state_tensor)
        return torch.argmax(Q_values, dim=1).item()

    def save(self: _ProjectAgent, path: str | None = None) -> None:
        """
        Save the agent.

        Args:
            self (_ProjectAgent): Class instance.
        """
        folder = os.path.join("src", "saved_models")
        os.makedirs(folder, exist_ok=True)
        torch.save(
            self.best_model.state_dict(),
            os.path.join(folder, f"agent_{self.id_experiment}.pth"),
        )

    def load(self: _ProjectAgent) -> None:
        """
        Load a pre-trained agent.

        Args:
            self (_ProjectAgent): Class isnatnce.
        """
        self.best_model.load_state_dict(
            torch.load(
                "best_model.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )


# if __name__ == "__main__":
#     from fast_env import FastHIVPatient
#     from gymnasium.wrappers import TimeLimit

#     agent = ProjectAgent()
#     agent.best_model = agent.model.train(
#         env=TimeLimit(
#             env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
#         )
#     )
#     agent.save()
