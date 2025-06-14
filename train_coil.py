import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal


class CoIL(nn.Module):
    def __init__(self, in_size, out_size):
        super(CoIL, self).__init__()

        # We define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2

        self.linear = nn.Linear(in_size, 128)
        self.linear_left = nn.Linear(128, out_size)
        self.linear_straight = nn.Linear(128, out_size)
        self.linear_right = nn.Linear(128, out_size)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear_left.weight)
        nn.init.xavier_uniform_(self.linear_straight.weight)
        nn.init.xavier_uniform_(self.linear_right.weight)

        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear_left.bias)
        nn.init.zeros_(self.linear_straight.bias)
        nn.init.zeros_(self.linear_right.bias)







    def forward(self, x, u):
        # We perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (?, |O|) tensor that keeps a batch of observations
        # - u is a (?, 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use



        x = self.linear(x)
        x = F.relu_(x)
        branch_left = (u == 0).reshape(-1)
        branch_straight = (u == 1).reshape(-1)
        branch_right = (u == 2).reshape(-1)

        input_left = x[branch_left,:]
        input_straight = x[branch_straight,:]
        input_right = x[branch_right,:]

        # Pass the intermediate result through branch-specific linear layers
        output_left = self.linear_left(input_left)
        output_straight = self.linear_straight(input_straight)
        output_right = self.linear_right(input_right)
        output = torch.zeros(x.shape[0], 2)
        # Combine the branch outputs using the branch selectors
        output[branch_left,:] = output_left
        output[branch_straight,:] = output_straight
        output[branch_right,:] = output_right
    


        return output


def run_training(data, args):
    """
    Trains a feedforward NN.
    """
    params = {
        "train_batch_size": 4096,
    }
    in_size = data["x_train"].shape[-1]
    out_size = data["y_train"].shape[-1]

    coil = CoIL(in_size, out_size)
    if args.restore:
        ckpt_path = (
            "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_CoIL"
        )
        coil.load_state_dict(torch.load(ckpt_path))

    optimizer = optim.Adam(coil.parameters(), lr=args.lr)

    def train_step(x, y, u):
        """
        We perform a single training step (for one batch):
        1. Make a forward pass through the model
        2. Calculate the loss for the output of the forward pass

        We want to compute the loss between y_est and y where
        - y_est is the output of the network for a batch of observations & goals,
        - y is the actions the expert took for the corresponding batch of observations & goals

    
        
        At the end your code should return the scalar loss value.
        HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
        """
        y_est = coil(x, u)
        loss = F.mse_loss(y_est, y)
        return loss

    dataset = TensorDataset(
        torch.Tensor(data["x_train"]),
        torch.Tensor(data["y_train"]),
        torch.Tensor(data["u_train"]),
    )
    dataloader = DataLoader(
        dataset, batch_size=params["train_batch_size"], shuffle=True
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for x, y, u in dataloader:
            optimizer.zero_grad()
            batch_loss = train_step(x, y, u)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        epoch_loss /= len(dataloader)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    ckpt_path = (
        "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_CoIL"
    )
    torch.save(coil.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--lr", type=float, help="learning rate for Adam optimizer", default=5e-3
    )
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = "all"

    maybe_makedirs("./policies")

    data = load_data(args)

    run_training(data, args)
