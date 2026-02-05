"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# TODO 硬编码参数，后续整理,目前实现的是libero架构
# from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8   

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        
        # [Robustness Fix] Run the first large LayerNorm in Float32 to prevent NaNs/Overflow with large input_dim (65536) in BFloat16
        # We manually apply F.layer_norm with casted weights/input
        x_float = x.float()
        w_float = self.layer_norm1.weight.float()
        b_float = self.layer_norm1.bias.float() if self.layer_norm1.bias is not None else None
        
        x = F.layer_norm(x_float, self.layer_norm1.normalized_shape, w_float, b_float, self.layer_norm1.eps)
        
        # Cast back to FC1 weight dtype (e.g. BFloat16) for the Linear layer
        target_dtype = self.fc1.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Ensure x matches fc1 dtype (sometimes weights stay in bfloat16 while inputs are float32 or vice versa)
        if x.dtype != self.fc1.weight.dtype:
            x = x.to(self.fc1.weight.dtype)
            
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=ground_truth_actions.shape, device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            timestep_embeddings=timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

class FlowMatchingActionHead(nn.Module):
    """
    MLP-based action head that generates continuous actions via flow matching.
    
    Flow matching is an alternative to diffusion models that directly learns the continuous-time flow
    between a simple distribution (e.g., standard normal) and the target distribution.
    
    Inspired by: https://arxiv.org/abs/2210.02747 and PI0: A Vision-Language-Action Flow Model
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.flow_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*action_dim, hidden_dim=hidden_dim, action_dim=action_dim
        )

        # Time encoder for positional encoding of timesteps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        
        # Projection for noisy actions to condition the hidden states
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def sample_noise(self, shape, device):
        """Sample noise from a standard normal distribution."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )
        return noise
    

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)
        
    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the flow prediction network. Returns noise, noisy actions, and the
        corresponding flow timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        
        # Ensure inputs match model dtype
        dtype = self.action_proj.weight.dtype
        if ground_truth_actions.dtype != dtype:
            ground_truth_actions = ground_truth_actions.to(dtype)

        
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        
        # Sample random noise with shape equal to actions
        noise = self.sample_noise(ground_truth_actions.shape, device).to(dtype)
        
        # Sample random flow timesteps (one for each action in batch)
        timesteps = self.sample_time(batch_size, device)
        
        # In flow matching, we interpolate between noise and ground truth based on timestep
        # x_t = (1-t) * noise + t * ground_truth
        time_expanded = timesteps.view(-1, 1, 1)
        noisy_actions = (1 - time_expanded) * noise + time_expanded * ground_truth_actions
        u_t = (ground_truth_actions - (1 - 0.001) * noise) / (1 - (1 - 0.001))
        # Optimal Transport Conditional Flow Matching target v_t = x_1 - (1-sigma_min)x_0
        # Here x_1 = ground_truth, x_0 = noise.
        # u_t = x_1 - (1 - 1e-5)*x_0 approx x_1 - x_0
        # Standard OT-CFM: u_t = x_1 - x_0.
        u_t = ground_truth_actions - (1 - 1e-4) * noise

        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
        
        return_dict = dict(
            noise=noise,
            flow=u_t,
            noisy_actions=noisy_actions,
            timestep_embeddings=timestep_embeddings,
        )
        
        return return_dict
    
    def predict_velocity(self, actions_hidden_states, noisy_actions, timestep_embeddings):
        """
        Given a batch of last hidden Transformer layer embeddings, noisy actions and time,
        predicts the velocity vector field.
        """
        # actions_hidden_states: (batch_size, chunk_len * action_dim, hidden_dim)
        # noisy_actions: (batch_size, chunk_len, action_dim)
        # timestep_embeddings: (batch_size, 1, hidden_dim)
        
        batch_size = actions_hidden_states.shape[0]
        chunk_len = noisy_actions.shape[1]
        
        # Project noisy actions: [B, Chunk, Dim] -> [B, Chunk, Hidden]
        action_emb = self.action_proj(noisy_actions)
        
        # Reshape VLM hidden states: [B, Chunk*Dim, Hidden] -> [B, Chunk, Dim, Hidden]
        reshaped_states = actions_hidden_states.reshape(batch_size, chunk_len, self.action_dim, self.hidden_dim)
        
        # Add conditioning information (broadcast over Dim dimension)
        # action_emb: [B, Chunk, 1, Hidden]
        # timestep_embeddings: [B, 1, 1, Hidden]
        
        conditioned_states = reshaped_states + action_emb.unsqueeze(2) + timestep_embeddings.unsqueeze(2)
        
        # Flatten back for the MLP: [B, Chunk, Dim*Hidden]
        flat_input = conditioned_states.reshape(batch_size, chunk_len, -1)
        
        # Predict
        velocity_pred = self.flow_predictor(flat_input)
        return velocity_pred