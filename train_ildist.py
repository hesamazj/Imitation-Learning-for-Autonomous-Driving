import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal


class MDN(nn.Module):
    def __init__(self, in_size, out_size):
        super(MDN, self).__init__()

        hidden_size = 64
        # We define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, out_size)
        self.cov_layer = nn.Linear(hidden_size, out_size * out_size)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.xavier_uniform_(self.cov_layer.weight)

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.zeros_(self.cov_layer.bias)



    def forward(self, x):
        # We perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?, |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        # IMPORTANT: The rest of the columns of the output tensor must correspond to the covariance matrix!
        # IMPORTANT: The covariance matrix must be symmetric and positive definite!
        # IMPORTANT: You can use torch.distributions.multivariate_normal.MultivariateNormal to create a multivariate normal distribution.

        temp = F.relu(self.linear1(x))
        mean = self.mean_layer(temp)
        temp_cov = self.cov_layer(temp)
        temp_cov = temp_cov.view(-1, 2, 2)
        temp_cov = torch.matmul(temp_cov, temp_cov.transpose(1, 2))
        temp_cov = temp_cov+torch.eye(2)*1e-3
        cov = temp_cov.view(-1, 4)

        y = torch.cat((mean, cov), 1)

        return y


def run_training(data, args):
    """
    Trains a feedforward NN.
    """
    params = {
        "train_batch_size": 4096 * 32,
    }
    in_size = data["x_train"].shape[-1]
    out_size = data["y_train"].shape[-1]

    mdn = MDN(in_size, out_size)
    if args.restore:
        ckpt_path = (
            "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_ILDIST"
        )
        mdn.load_state_dict(torch.load(ckpt_path))

    optimizer = optim.Adam(mdn.parameters(), lr=args.lr)

    def train_step(x, y):
        """
        We perform a single training step (for one batch):
        1. Make a forward pass through the model
        2. Calculate the loss for the output of the forward pass

        We want to compute the negative log-likelihood loss between y_est and y where
        - y_est is the output of the network for a batch of observations,
        - y is the actions the expert took for the corresponding batch of observations
        At the end your code should return the scalar loss value.
        """
        y_est = mdn(x)
        log_prob = torch.distributions.multivariate_normal.MultivariateNormal(y_est[:, :2], y_est[:, 2:].view(-1,2,2)).log_prob(y)
        loss = -torch.mean(log_prob)
        return loss

    # load dataset
    dataset = TensorDataset(
        torch.Tensor(data["x_train"]), torch.Tensor(data["y_train"])
    )
    dataloader = DataLoader(
        dataset, batch_size=params["train_batch_size"], shuffle=True
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for x, y in dataloader:
            optimizer.zero_grad()
            batch_loss = train_step(x, y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        epoch_loss /= len(dataloader)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    ckpt_path = (
        "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_ILDIST"
    )
    torch.save(mdn.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="left, straight, right, inner, outer, all",
        default="all",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="intersection, circularroad",
        default="intersection",
    )
    parser.add_argument(
        "--epochs", type=int, help="number of epochs for training", default=1000
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate for Adam optimizer", default=1e-3
    )
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()

    maybe_makedirs("./policies")

    data = load_data(args)

    run_training(data, args)
