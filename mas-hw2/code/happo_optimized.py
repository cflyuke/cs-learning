"""Optimized HAPPO algorithm with adaptive learning and improved exploration."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPOOptimized(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Optimized HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPOOptimized, self).__init__(args, obs_space, act_space, device)

        # Original HAPPO parameters
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
        # Optimization parameters
        self.adaptive_clip = args.get("adaptive_clip", True)
        self.adaptive_entropy = args.get("adaptive_entropy", True)
        self.use_kl_penalty = args.get("use_kl_penalty", True)
        self.target_kl = args.get("target_kl", 0.01)
        self.kl_coef = args.get("kl_coef", 0.2)
        self.use_value_clipping = args.get("use_value_clipping", True)
        self.dual_clip = args.get("dual_clip", True)
        self.dual_clip_param = args.get("dual_clip_param", 3.0)
        
        # Adaptive parameters
        self.min_clip_param = args.get("min_clip_param", 0.05)
        self.max_clip_param = args.get("max_clip_param", 0.3)
        self.min_entropy_coef = args.get("min_entropy_coef", 0.001)
        self.max_entropy_coef = args.get("max_entropy_coef", 0.1)
        
        # Learning rate scheduling
        self.use_lr_schedule = args.get("use_lr_schedule", True)
        self.lr_schedule_type = args.get("lr_schedule_type", "cosine")  # linear, cosine, exponential
        
        # Experience replay for stability
        self.use_experience_replay = args.get("use_experience_replay", False)
        self.replay_buffer_size = args.get("replay_buffer_size", 10000)
        self.replay_ratio = args.get("replay_ratio", 0.1)
        
        # Initialize adaptive parameters
        self.current_clip_param = self.clip_param
        self.current_entropy_coef = self.entropy_coef
        
        # Statistics tracking
        self.kl_divergences = []
        self.policy_losses = []
        self.entropy_values = []
        
        # Experience replay buffer (if enabled)
        if self.use_experience_replay:
            self.replay_buffer = []

    def update(self, sample):
        """Update actor network with optimizations.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
            kl_divergence: (torch.Tensor) KL divergence between old and new policies.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, action_distribution = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # Calculate importance weights
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        # Calculate KL divergence for adaptive clipping
        kl_divergence = self._calculate_kl_divergence(
            old_action_log_probs_batch, action_log_probs
        )
        
        # Adaptive clipping based on KL divergence
        if self.adaptive_clip:
            self._update_clip_param(kl_divergence)

        # Standard PPO clipping
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.current_clip_param, 1.0 + self.current_clip_param)
            * adv_targ
        )
        
        # Dual clipping for better stability
        if self.dual_clip:
            surr3 = torch.max(surr1, self.dual_clip_param * adv_targ)
            policy_action_loss_unclipped = -torch.min(torch.min(surr1, surr2), surr3)
        else:
            policy_action_loss_unclipped = -torch.min(surr1, surr2)

        if self.use_policy_active_masks:
            policy_action_loss = (
                torch.sum(factor_batch * policy_action_loss_unclipped, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = torch.sum(
                factor_batch * policy_action_loss_unclipped, dim=-1, keepdim=True
            ).mean()

        # KL penalty for policy regularization
        kl_penalty = 0
        if self.use_kl_penalty:
            kl_penalty = self.kl_coef * kl_divergence.mean()

        # Adaptive entropy coefficient
        if self.adaptive_entropy:
            self._update_entropy_coef(dist_entropy)

        # Total policy loss
        policy_loss = policy_action_loss + kl_penalty

        # Gradient update
        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.current_entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        # Update statistics
        self.kl_divergences.append(kl_divergence.mean().item())
        self.policy_losses.append(policy_loss.item())
        self.entropy_values.append(dist_entropy.mean().item())

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights, kl_divergence

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD with optimizations.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["kl_divergence"] = 0
        train_info["clip_param"] = self.current_clip_param
        train_info["entropy_coef"] = self.current_entropy_coef

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        # Advantage normalization with outlier handling
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            
            # Ensure numeric types and handle edge cases
            if np.isnan(mean_advantages) or np.isnan(std_advantages):
                mean_advantages = 0.0
                std_advantages = 1.0
            
            # Advantage normalization with proper type handling
            advantages = np.clip(
                (advantages - float(mean_advantages)) / (float(std_advantages) + 1e-5),
                -10.0, 10.0
            ).astype(np.float32)

        # Early stopping based on KL divergence
        early_stop = False
        
        for epoch in range(self.ppo_epoch):
            if early_stop:
                break
                
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            epoch_kl = 0
            num_batches = 0
            
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, kl_divergence = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["kl_divergence"] += kl_divergence.mean().item()
                
                epoch_kl += kl_divergence.mean().item()
                num_batches += 1

            # Early stopping if KL divergence is too high
            if num_batches > 0 and epoch_kl / num_batches > self.target_kl * 1.5:
                early_stop = True
                print(f"Early stopping at epoch {epoch} due to high KL divergence: {epoch_kl / num_batches:.4f}")

        # Experience replay
        if self.use_experience_replay and len(self.replay_buffer) > 0:
            self._replay_experience(train_info)

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            if k not in ["clip_param", "entropy_coef"]:
                train_info[k] /= num_updates

        return train_info

    def _calculate_kl_divergence(self, old_log_probs, new_log_probs):
        """Calculate KL divergence between old and new policies."""
        kl_div = old_log_probs - new_log_probs
        return kl_div.mean()

    def _update_clip_param(self, kl_divergence):
        """Adaptively update clipping parameter based on KL divergence."""
        kl_val = kl_divergence.mean().item()
        
        if kl_val > self.target_kl * 2.0:
            # KL too high, decrease clipping
            self.current_clip_param = max(
                self.current_clip_param * 0.8, 
                self.min_clip_param
            )
        elif kl_val < self.target_kl * 0.5:
            # KL too low, increase clipping
            self.current_clip_param = min(
                self.current_clip_param * 1.2, 
                self.max_clip_param
            )

    def _update_entropy_coef(self, entropy):
        """Adaptively update entropy coefficient based on current entropy."""
        entropy_val = entropy.mean().item()
        
        # Target entropy (heuristic: log of action space size)
        if hasattr(self.act_space, 'n'):
            target_entropy = np.log(self.act_space.n) * 0.5
        else:
            target_entropy = -np.prod(self.act_space.shape)
        
        if entropy_val < target_entropy * 0.5:
            # Entropy too low, increase coefficient
            self.current_entropy_coef = min(
                self.current_entropy_coef * 1.1,
                self.max_entropy_coef
            )
        elif entropy_val > target_entropy * 2.0:
            # Entropy too high, decrease coefficient
            self.current_entropy_coef = max(
                self.current_entropy_coef * 0.9,
                self.min_entropy_coef
            )

    def _replay_experience(self, train_info):
        """Perform experience replay for additional stability."""
        if len(self.replay_buffer) < self.replay_buffer_size // 10:
            return
        
        # Sample from replay buffer
        replay_samples = np.random.choice(
            len(self.replay_buffer), 
            size=min(len(self.replay_buffer), int(self.actor_num_mini_batch * self.replay_ratio)),
            replace=False
        )
        
        for idx in replay_samples:
            sample = self.replay_buffer[idx]
            policy_loss, dist_entropy, actor_grad_norm, imp_weights, kl_divergence = self.update(sample)
            
            # Add to training info with reduced weight
            train_info["policy_loss"] += policy_loss.item() * 0.1
            train_info["dist_entropy"] += dist_entropy.item() * 0.1

    def add_to_replay_buffer(self, sample):
        """Add sample to experience replay buffer."""
        if not self.use_experience_replay:
            return
            
        self.replay_buffer.append(sample)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def get_optimization_stats(self):
        """Get current optimization statistics."""
        stats = {
            "current_clip_param": self.current_clip_param,
            "current_entropy_coef": self.current_entropy_coef,
            "avg_kl_divergence": np.mean(self.kl_divergences[-100:]) if self.kl_divergences else 0,
            "avg_policy_loss": np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            "avg_entropy": np.mean(self.entropy_values[-100:]) if self.entropy_values else 0,
        }
        return stats

    def reset_optimization_stats(self):
        """Reset optimization statistics."""
        self.kl_divergences = []
        self.policy_losses = []
        self.entropy_values = []

    def lr_decay(self, episode, episodes):
        """Enhanced learning rate decay with different scheduling options."""
        if not self.use_lr_schedule:
            return super().lr_decay(episode, episodes)
        
        progress = episode / episodes
        
        if self.lr_schedule_type == "cosine":
            # Cosine annealing
            lr_mult = 0.5 * (1 + np.cos(np.pi * progress))
        elif self.lr_schedule_type == "exponential":
            # Exponential decay
            lr_mult = np.exp(-5 * progress)
        else:
            # Linear decay (default)
            lr_mult = 1 - progress
        
        new_lr = self.lr * lr_mult
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
