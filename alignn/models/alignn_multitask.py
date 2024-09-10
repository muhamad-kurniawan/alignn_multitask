"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from typing import Literal, List, Sequence
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


# class ALIGNNConfig(BaseSettings):
#     """Hyperparameter schema for jarvisdgl.models.alignn."""

#     name: Literal["alignn"]
#     alignn_layers: int = 4
#     gcn_layers: int = 4
#     atom_input_features: int = 92
#     edge_input_features: int = 80
#     triplet_input_features: int = 40
#     embedding_features: int = 64
#     hidden_features: int = 256
#     # fc_layers: int = 1
#     # fc_features: int = 64
#     output_features: int = 1

#     # if link == log, apply `exp` to final outputs
#     # to constrain predictions to be positive
#     link: Literal["identity", "log", "logit"] = "identity"
#     zero_inflated: bool = False
#     classification: bool = False
#     num_classes: int = 2
#     extra_features: int = 0

#     class Config:
#         """Configure model settings behavior."""

#         env_prefix = "jv_model"

class ALIGNNMTConfig(BaseSettings):
    """Configuration for each task in the multitask model."""
    name: Literal["alignnmt"]
    task_types: List[Literal["regression", "classification"]] = ["regression"]
    output_nodes: List[int] = [1] # This list should contain the number of output nodes per task
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    extra_features: int = 0
    robust: bool = False
    
    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNNMT(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNMTConfig = ALIGNNMTConfig(name="alignnmt")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()
        # if self.classification:
        #     self.fc = nn.Linear(config.hidden_features, config.num_classes)
        #     self.softmax = nn.LogSoftmax(dim=1)
        # else:
        
        # self.fc = nn.Linear(config.hidden_features, config.output_features)

        if config.extra_features != 0:
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        # self.link = None
        # self.link_name = config.link
        # if config.link == "identity":
        #     self.link = lambda x: x
        # elif config.link == "log":
        #     self.link = torch.exp
        #     avg_gap = 0.7  # magic number -- average bandgap in dft_3d
        #     self.fc.bias.data = torch.tensor(
        #         np.log(avg_gap), dtype=torch.float
        #     )
        # elif config.link == "logit":
        #     self.link = torch.sigmoid

        self.heads = nn.ModuleList()
        self.normalizers = {}
        # for task_type, output_nodes in zip(config.task_types, config.output_nodes):
        #     if task_type == "regression":
        #         self.heads.append(nn.Linear(config.hidden_features, output_nodes))
        #     elif task_type == "classification":
        #         if output_nodes == 2:  # Binary classification
        #             self.heads.append(nn.Sequential(
        #                 nn.Linear(config.hidden_features, 1),
        #                 nn.BCEWithLogitsLoss()
        #             ))
        #         else:  # Multi-class classification
        #             self.heads.append(nn.Sequential(
        #                 nn.Linear(config.hidden_features, output_nodes),
        #                 nn.CrossEntropyLoss(dim=-1)
        #             ))
        if self.config.robust:
            output_nodes = [2 * nodes for nodes in sum(self.config.output_nodes)]
        else:
            output_nodes = self.config.output_nodes

        self.heads = nn.ModuleList(
            ResidualNetwork(
                input_dim=config.hidden_features,   # Input from the hidden layer
                output_dim=nodes,        # 2x output for mean and log_std
                hidden_layer_dims=[64, 64],         # Example hidden layers
                activation=nn.ReLU,                # Activation function
                batch_norm=True                     # Use batch normalization
            ) for nodes in output_nodes
        )
        for idx, (task_type, output_nodes) in enumerate(zip(config.task_types, config.output_nodes)):
            if task_type == "regression":
                self.normalizers[task_type] = Normalizer()


    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            # print('features2',features.shape)

            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('g',g)
            # print('features1',features.shape)
            features = self.extra_feature_embedding(features)

        g = g.local_var()
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        # print('x1',x.shape)
        x = self.atom_embedding(x)
        # print('x2',x.shape)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        # print('h',h.shape)
        # print('features',features.shape)
        if self.config.extra_features != 0:
            h_feat = self.readout_feat(g, features)
            # print('h1',h.shape)
            # print('h_feat',h_feat.shape)
            h = torch.cat((h, h_feat), 1)
            # print('h2',h.shape)

            h = self.fc1(h)

            h = self.fc2(h)

            out = self.fc3(h)
        # else:
        #     out = self.fc(h)

        # if self.link:
        #     out = self.link(out)

        # if self.classification:
        #     # out = torch.round(torch.sigmoid(out))
        #     out = self.softmax(out)

        outputs = []
        # for head in self.heads:
            
        #     outputs.append(head(h))
        if self.config.robust:
            for idx, head in enumerate(self.heads):
                output = head(h)  # Output will be (2 * output_nodes,)
                pred_mean, pred_log_std = torch.chunk(output, 2, dim=-1)  # Split into mean and log_std
                outputs.append((pred_mean, pred_log_std))
        else:
            for head in self.heads:
                outputs.append(head(h))

        # return torch.squeeze(out)
        return outputs

    # def save(self, path: str):
    #     """Save model parameters and normalizer states to the specified path."""
    #     model_state = {
    #         "model_state_dict": self.state_dict(),
    #         "normalizers": {f"normalizer_{idx}": normalizer.state_dict() for idx, normalizer in self.normalizers.items()}
    #     }
    #     torch.save(model_state, path)

    # def load(self, path: str):
    #     """Load model parameters and normalizer states from the specified path."""
    #     model_state = torch.load(path)
    #     self.load_state_dict(model_state["model_state_dict"])
    #     for idx, normalizer_state in model_state["normalizers"].items():
    #         self.normalizers[idx].load_state_dict(normalizer_state)

    def save_state(self, filepath):
        """Save the model and normalizers."""
        # Save the model state_dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'normalizers': [norm.state_dict() for norm in self.normalizers]
        }, filepath)

    def load_state(self, filepath):
        """Load the model and normalizers."""
        # Load the state from the file
        checkpoint = torch.load(filepath)

        # Load the model state_dict
        self.load_state_dict(checkpoint['model_state_dict'])

        # Load normalizer states
        normalizer_states = checkpoint['normalizers']
        for norm, state in zip(self.normalizers, normalizer_states):
            norm.load_state_dict(state)

class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self) -> None:
        """Initialize Normalizer with mean 0 and std 1."""
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor, dim: int = 0, keepdim: bool = False) -> None:
        """Compute the mean and standard deviation of the given tensor.

        Args:
            tensor (Tensor): Tensor to determine the mean and standard deviation over.
            dim (int, optional): Which dimension to take mean and standard deviation
                over. Defaults to 0.
            keepdim (bool, optional): Whether to keep the reduced dimension in Tensor.
                Defaults to False.
        """
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        """Normalize a Tensor.

        Args:
            tensor (Tensor): Tensor to be normalized

        Returns:
            Tensor: Normalized Tensor
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """Restore normalized Tensor to original.

        Args:
            normed_tensor (Tensor): Tensor to be restored

        Returns:
            Tensor: Restored Tensor
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        """Get Normalizer parameters mean and std.

        Returns:
            dict[str, Tensor]: Dictionary storing Normalizer parameters.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict) -> None:
        """Overwrite Normalizer parameters given a new state_dict.

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.
        """
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls):
        """Create a new Normalizer given a state_dict.

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.

        Returns:
            Normalizer
        """
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


def sampled_softmax(pre_logits, log_std, samples: int = 10):
    """Draw samples from Gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.

    Args:
        pre_logits (Tensor): Expected logits before softmax.
        log_std (Tensor): Deviation in logits before softmax.
        samples (int, optional): Number of samples to take. Defaults to 10.

    Returns:
        Tensor: Averaged logits sampled from pre-logits
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = torch.exp(log_std).repeat_interleave(samples, dim=0)

    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + torch.mul(
        epsilon, sam_std
    )
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return torch.mean(logits, dim=1)

class ResidualNetwork(nn.Module):
    """Feed forward Residual Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,
    ) -> None:
        """Create a feed forward neural network with skip connections.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(
                nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
            )
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.res_fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=False)
            if (dims[idx] != dims[idx + 1])
            else nn.Identity()
            for idx in range(len(dims) - 1)
        )
        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        """Forward pass through network."""
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dim = self.fc_out.out_features
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dim=}, {activation=})"
