import torch
from model import Actor, Critic


class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.critic.parameters()},
        ], lr=cfg.lr)

        # rollout buffer
        self.obs_buf = torch.zeros(cfg.horizon, cfg.num_envs, obs_dim, device=self.device)
        self.act_buf = torch.zeros(cfg.horizon, cfg.num_envs, act_dim, device=self.device)
        self.logp_buf = torch.zeros(cfg.horizon, cfg.num_envs, device=self.device)
        self.rew_buf = torch.zeros(cfg.horizon, cfg.num_envs, device=self.device)
        self.done_buf = torch.zeros(cfg.horizon, cfg.num_envs, device=self.device)
        self.val_buf = torch.zeros(cfg.horizon, cfg.num_envs, device=self.device)

    @torch.no_grad()
    def collect_rollout(self, env, obs, obs_rms=None):
        for t in range(self.cfg.horizon):
            # normalize obs before feeding to policy
            obs_input = obs_rms.normalize(obs) if obs_rms else obs

            action, log_prob, _ = self.actor.get_action(obs_input)
            value = self.critic(obs_input)

            next_obs, reward, done, info = env.step(action)

            self.obs_buf[t] = obs_input
            self.act_buf[t] = action
            self.logp_buf[t] = log_prob
            self.rew_buf[t] = reward
            self.done_buf[t] = done.float()
            self.val_buf[t] = value

            obs = next_obs

        return obs

    @torch.no_grad()
    def compute_gae(self, last_obs):
        last_val = self.critic(last_obs)
        advantages = torch.zeros_like(self.rew_buf)

        gae = torch.zeros(self.cfg.num_envs, device=self.device)
        for t in reversed(range(self.cfg.horizon)):
            if t == self.cfg.horizon - 1:
                next_val = last_val
            else:
                next_val = self.val_buf[t + 1]

            not_done = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + self.cfg.gamma * next_val * not_done - self.val_buf[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + self.val_buf
        return advantages, returns

    def update(self, advantages, returns):
        total = self.cfg.horizon * self.cfg.num_envs
        obs_flat = self.obs_buf.reshape(total, -1)
        act_flat = self.act_buf.reshape(total, -1)
        logp_old = self.logp_buf.reshape(total)
        adv_flat = advantages.reshape(total)
        ret_flat = returns.reshape(total)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.cfg.epochs_per_update):
            indices = torch.randperm(total, device=self.device)

            for start in range(0, total, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                idx = indices[start:end]

                mb_obs = obs_flat[idx]
                mb_act = act_flat[idx]
                mb_logp_old = logp_old[idx]
                mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]

                # policy loss
                logp_new = self.actor.get_log_prob(mb_obs, mb_act)
                ratio = (logp_new - mb_logp_old).exp()
                clipped = ratio.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()

                # value loss
                values = self.critic(mb_obs)
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                # entropy bonus
                _, _, entropy = self.actor.get_action(mb_obs)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        num_updates = self.cfg.epochs_per_update * (total // self.cfg.minibatch_size)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
        }