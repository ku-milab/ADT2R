import numpy as np
import torch.nn as nn
import torch, math
import torch.nn.functional as F
from copy import deepcopy


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device, n_tokens):
        super().__init__()

        self.device = device
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.n_tokens = n_tokens

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        # self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        self.proj_nets = nn.ModuleList([nn.Linear(h_dim, h_dim) for i in range(self.n_heads)])

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, (self.h_dim // self.n_heads) * self.n_tokens))

        self.mse_none = nn.MSELoss(reduction="none")

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def orthogonal_constraint(self, x):
        B, D1, D2 = x.shape

        normalized_x = F.normalize(x, p=2, dim=-1)
        factor_transpose = normalized_x.transpose(2, 1)
        multiplied = torch.bmm(normalized_x, factor_transpose)
        identity = torch.eye(D1).reshape(1, D1, D1).repeat(B, 1, 1).to(self.device)
        mse = self.mse_none(input=multiplied, target=identity)
        frobenius = torch.norm(mse, p="fro", dim=[-2, -1])

        return frobenius.mean()  # Taken an average along batch dimension

    def forward(self, x, sindices=None):

        B, T, H = x.shape
        N, D = self.n_heads, H // self.n_heads

        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)

        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))

        normalized_weights = F.softmax(weights, dim=-1)

        attention_ = self.att_drop(normalized_weights @ v)

        attention_flatten = attention_.reshape([B, N, -1])
        loss_reg = self.orthogonal_constraint(attention_flatten)

        attention = attention_.transpose(1, 2)

        if sindices == None:
            cluster_idx_ = self.avg_pool(
                attention_.reshape(B, N, int(self.max_T / self.n_tokens), -1).transpose(1, 2))
            cluster_idx_ = torch.argmax(torch.softmax(cluster_idx_.squeeze(dim=-1), dim=-1), dim=-1)
            cluster_idx = cluster_idx_.unsqueeze(dim=2).repeat(1, 1, self.n_tokens).reshape(B, -1)

        else:
            cluster_idx_ = sindices
            cluster_idx = sindices.unsqueeze(dim=2).repeat(1, 1, self.n_tokens).reshape(B, -1)

        attention = attention.contiguous().view(B, T, N * D)

        outs = [self.proj_nets[i](attention) for i in range(self.n_heads)]
        outs = torch.stack(outs, dim=-1)  #
        outs = outs.view(-1, outs.shape[2], outs.shape[-1])
        final_outs = outs[list(range(len(outs))), :, cluster_idx.view(-1)]
        final_outs = final_outs.view(B, T, -1)

        return loss_reg, final_outs, cluster_idx_

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device, n_tokens):
        super().__init__()

        self.n_heads = n_heads
        self.h_dim = h_dim
        self.max_t = max_T
        self.drop_p = drop_p
        self.device = device
        self.n_tokens = n_tokens

        self.attention = MaskedCausalAttention(self.h_dim, self.max_t, self.n_heads, self.drop_p, self.device, self.n_tokens)
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        ) for i in range(self.n_heads)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(h_dim) for i in range(self.n_heads)])
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, sindices=None):

        B, T, H = x.shape

        loss_reg, attended, cidx = self.attention(x, sindices)
        x = x + attended
        x = [self.ln1[i](x) for i in range(self.n_heads)]
        x = [x[i] + self.mlp[i](x[i]) for i in range(self.n_heads)]
        x_ = torch.stack(x, dim=-1)
        x_ = x_.view(-1, x_.shape[2], x_.shape[-1])
        cidx_ = cidx.unsqueeze(dim=2).repeat(1, 1, self.n_tokens).reshape(B, -1)
        final = x_[list(range(len(x_))), :, cidx_.view(-1)]
        final = final.view(B, T, -1)
        x = self.ln2(final)

        return x, loss_reg, cidx

class ADT2R(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim, n_heads, drop_p, max_t, device, lr, w_decay, lr_decay, lr_step, lam_actor, lam_critic, lam_reg, gamma, tau):
        super().__init__()

        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.max_t = max_t
        self.lr = lr
        self.w_decay = w_decay
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self.lam_reg = lam_reg
        self.gamma = gamma
        self.tau = tau

        self.embed_ln = nn.LayerNorm(self.h_dim)
        self.embed_timestep = nn.Embedding(self.max_t, h_dim)
        self.embed_state = nn.Linear(self.state_dim, self.h_dim)
        self.embed_action = nn.Embedding(self.act_dim, self.h_dim)
        self.embed_mortality = nn.Linear(1, self.h_dim)
        self.embed_estimated_state = nn.Linear(1, self.h_dim)
        self.embed_hiddens_low= nn.Sequential(torch.nn.Linear(h_dim * 2, h_dim),
                                               nn.LayerNorm(h_dim),
                                               nn.GELU(),
                                               nn.Dropout())

        self.critic = nn.Linear(self.h_dim, self.act_dim)
        self.critic_target = deepcopy(self.critic)
        self.actor = nn.Linear(self.h_dim, self.act_dim)
        self.actor_target = deepcopy(self.actor)

        self.ve_adt = Block(self.h_dim, self.max_t * 2, self.n_heads, self.drop_p, self.device, self.n_tokens)

        self.tr_adt = Block(self.h_dim, self.max_t * 3, self.n_heads, self.drop_p, self.device, self.n_tokens)

        self.embed_ln_tr= nn.LayerNorm(self.h_dim)

        self.policy = nn.Linear(h_dim, act_dim)


        self.optimiser_actor = torch.optim.RAdam(
            list(self.embed_ln.parameters()) + list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters())
            + list(self.ve_adt.parameters()) + list(self.actor.parameters()),
            lr=self.lr, weight_decay=self.w_decay)

        self.optimiser_critic = torch.optim.RAdam(
            list(self.embed_ln.parameters()) + list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters())
            + list(self.ve_adt.parameters()) + list(self.critic.parameters()),
            lr=self.lr, weight_decay=self.w_decay)

        self.optimiser_all = torch.optim.RAdam(
            list(self.embed_ln.parameters()) + list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters())
            + list(self.tr_adt.parameters()) + list(
                self.embed_mortality.parameters())
            + list(self.embed_estimated_state.parameters()) + list(
                self.policy.parameters()),
            lr=self.lr, weight_decay=self.w_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser_all, gamma=self.lr_decay,
                                                         step_size=self.lr_step)

        self.ce_none = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def forward(self, records, is_train=True):

        timesteps = records["sequences"].type(torch.LongTensor).to(self.device)
        states = records["observation"]
        actions = records["action"]
        actions_ = actions.view(-1)
        demos = records["demo"]

        SRES = records["sofa_res"]  # [B,T]
        SCOA = records["sofa_coa"]
        SLIV = records["sofa_liv"]
        SCAR = records["sofa_car"]
        SCNS = records["sofa_cns"]
        SREN = records["sofa_ren"]

        SALL = torch.cat((SRES.unsqueeze(dim=2) / 4, SCOA.unsqueeze(dim=2) / 4, SLIV.unsqueeze(dim=2) / 4,
                          SCAR.unsqueeze(dim=2) / 4, SCNS.unsqueeze(dim=2) / 4, SREN.unsqueeze(dim=2) / 4),
                         dim=-1)  # [B, T, 6]

        states = torch.cat((states, SALL, demos), dim=-1)

        B, T, input_dim = states.shape

        """ Token Embedding module """
        time_embeddings = self.embed_timestep(timesteps)  # [B,T,H]
        state_embeddings = self.embed_state(states) + time_embeddings  # [B,T,H]
        action_embeddings = self.embed_action(actions) + time_embeddings  # [B,T,H]
        state_action_embeddings = torch.stack((state_embeddings, action_embeddings), dim=1)  # [B,2,T,H]
        state_action_embeddings = state_action_embeddings.permute(0, 2, 1, 3)  # [B,T,2,H]
        state_action_embeddings = state_action_embeddings.reshape(B, 2 * T, self.h_dim)
        ve_stacked_embeddings = self.embed_ln(state_action_embeddings)  # [B, 2*T, H]

        """ Value Estimation module """
        h, loss_reg, cidx = self.ve_adt(ve_stacked_embeddings)

        h_state_action = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)
        h_state = h_state_action[:,0]

        v_hat = self.critic(h_state)
        v_size = list(v_hat.size())  # [B,T,C]
        v_hat_ = v_hat.view(-1, self.act_dim)  # [B*T, 25]
        v_hat_at = v_hat_[list(range(v_size[0] * v_size[1])), list(actions_.data.cpu().numpy().astype(np.int32))].view(
            v_size[:2])  # [B,T]
        # Here, list(actions.data~~) is the action class index.

        v_hat_next = self.critic_target(h_state)
        v_hat_next = torch.cat((v_hat_next[:, 1:], torch.zeros(size=(v_size[0], 1, v_size[-1]), device=self.device)),
                               dim=1)  # [B,T,C]
        v_hat_next = v_hat_next.view(-1, self.act_dim)

        a_hat_target = torch.argmax(F.softmax(self.actor_target(h_state), dim=-1), dim=-1)  # [B, T]
        a_hat_target_next = torch.cat((a_hat_target[:, 1:], torch.zeros(size=(a_hat_target.shape[0], 1), device=self.device)), dim=-1).view(-1)  # [B,T]
        v_hat_at_next = v_hat_next[
            list(range(v_size[0] * v_size[1])), list(a_hat_target_next.data.cpu().numpy().astype(np.int32))].view(
            v_size[:2])  # [B,T]

        # Real reward
        R_T = records["mortality"] * 15
        # If the patient at the last step survived: 15, died: -15, otherwise: 0.

        next = R_T + self.gamma * v_hat_at_next

        td = v_hat_at - next  # [B,T]

        loss_critic = td ** 2 * records["seq_mask"]  # [B,T]
        loss_critic = self.lam_critic * loss_critic.sum() / records["seq_mask"].sum()

        if is_train: # Critic
            self.optimiser_critic.zero_grad()
            loss_critic.backward()
            self.optimiser_critic.step()

        time_embeddings = self.embed_timestep(timesteps)  # [B,T,H]
        state_embeddings = self.embed_state(states) + time_embeddings  # [B,T,H]
        action_embeddings = self.embed_action(actions) + time_embeddings  # [B,T,H]
        state_action_embeddings = torch.stack((state_embeddings, action_embeddings), dim=1)  # [B,2,T,H]
        state_action_embeddings = state_action_embeddings.permute(0, 2, 1, 3)  # [B,T,2,H]
        state_action_embeddings = state_action_embeddings.reshape(B, 2 * T, self.h_dim)
        stacked_embeddings = self.embed_ln(state_action_embeddings)  # [B, 2*T, H]

        h, loss_reg, cidx = self.transformer(stacked_embeddings)
        h_state_action = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)  # [B,2,T,H]
        h_state = h_state_action[:, 0]  # [B,T,H]


        actor_logit = self.actor(h_state)  # [B,T,C]
        actor_prob = torch.softmax(actor_logit, dim=-1)  # [B,T,C]
        actor_pred = torch.argmax(actor_prob, dim=-1)  # [B,T]
        log_prob = torch.log(actor_prob + 1e-5)
        log_prob_a = log_prob.view(-1, v_size[2])[
            list(range(v_size[0] * v_size[1])), list(actions_.data.cpu().numpy().astype(np.int32))].view(
            v_size[:2])  # [B*T] -> [B, T]
        loss_actor = -log_prob_a * v_hat_at.detach() * records["seq_mask"]  # [B,T]
        loss_actor = self.lam_actor * loss_actor.sum() / records["seq_mask"].sum()

        if is_train: # Actor
            self.optimiser_actor.zero_grad()
            loss_actor.backward()
            self.optimiser_actor.step()

            ADT2R.soft_update(self.critic, self.critic_target, self.tau)
            ADT2R.soft_update(self.actor, self.actor_target, self.tau)


        """ Treatment Recommendation module """
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        state_action_embeddings = torch.stack((state_embeddings, action_embeddings), dim=1)
        state_action_embeddings = state_action_embeddings.permute(0, 2, 1, 3)
        state_action_embeddings = state_action_embeddings.reshape(B, 2 * T, self.h_dim)
        stacked_embeddings = self.embed_ln(state_action_embeddings)

        h_low, loss_reg, cidx = self.transformer(stacked_embeddings)

        h_state_action = h_low.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)
        h_state = h_state_action[:, 0]

        v_hat = self.critic(h_state)
        v_hat_ = v_hat.view(-1, self.act_dim)
        v_hat_at = v_hat_[list(range(v_size[0] * v_size[1])), list(actions_.data.cpu().numpy().astype(np.int32))].view(
            v_size[:2])  # [B,T]
        # Here, list(actions.data~~) is the action class index.

        if is_train:
            goal_embeddings = self.embed_mortality(records["mortality_last"].unsqueeze(dim=1))
        else:
            goal_embeddings = self.embed_mortality(torch.ones(size=(B, 1, 1), device=self.device))

        zero_value = torch.zeros(size=(B, 1), device=self.device)
        v_hats = torch.cat((zero_value, v_hat_at[:, :-1]), dim=1)
        v_hats_embeddings = self.embed_estimated_state(v_hats.unsqueeze(dim=2).detach()) + time_embeddings
        goal_action_embeddings = torch.cat((goal_embeddings, action_embeddings[:, :-1, :]), dim=1)
        goal_action_value_state_embeddings = torch.stack((goal_action_embeddings, v_hats_embeddings, state_embeddings),
                                                         dim=1)
        goal_action_value_state_embeddings = goal_action_value_state_embeddings.permute(0, 2, 1, 3)
        goal_action_value_state_embeddings = goal_action_value_state_embeddings.reshape(B, 3 * T, self.h_dim)
        high_stacked_embeddings = self.embed_ln_high(goal_action_value_state_embeddings)

        h_hat, loss_reg_hat, cidx_hat = self.transformer_high(
            high_stacked_embeddings, sindices=cidx)
        h_goal_action_value_state_high = h_hat.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        h_state_high = h_goal_action_value_state_high[:, 2]

        action_logits = self.predict_action(h_state_high)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_preds = torch.argmax(action_probs, dim=-1)

        action_loss = self.ce_none(input=action_logits.view(-1, self.act_dim), target=actions.view(-1)) * records[
            "seq_mask"].view(-1)
        action_loss = action_loss.sum() / records["seq_mask"].view(-1).sum()

        reg_loss = self.lam_reg * (loss_reg + loss_reg_hat)
        action_reg_loss = action_loss + reg_loss

        if is_train:
            self.optimiser_all.zero_grad()
            action_reg_loss.backward()
            self.optimiser_all.step()

        return action_loss, reg_loss, loss_actor, loss_critic, action_probs, action_preds
