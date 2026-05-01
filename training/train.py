import torch
import torch.nn.functional as F
from yaml import safe_load
from types import SimpleNamespace

config = safe_load(open('config/config.yaml', 'r'))


def dict_to_obj(d):
    return SimpleNamespace(**{k: dict_to_obj(v) if isinstance(v, dict) else v for k, v in d.items()})


config = dict_to_obj(config)


def calc_loss(batch, net, tgt_net, device="cuda"):
    states, actions, rewards, next_states, dones, _, weights = batch

    states_v = torch.from_numpy(states).to(device, non_blocking=True).float() / 255.0
    next_states_v = torch.from_numpy(next_states).to(device, non_blocking=True).float() / 255.0
    actions_v = torch.from_numpy(actions).to(device, non_blocking=True).long()
    rewards_v = torch.from_numpy(rewards).to(device, non_blocking=True).float()
    done_mask = torch.from_numpy(dones.astype('bool')).to(device, non_blocking=True)
    weights_v = torch.from_numpy(weights).to(device, non_blocking=True)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_actions = net(next_states_v).argmax(1, keepdim=True)
        next_state_values = tgt_net(next_states_v).gather(1, next_actions).squeeze(-1)
        next_state_values[done_mask] = 0.0
    expected_state_action_values = rewards_v + (config.agent.gamma ** config.agent.n_steps) * next_state_values

    td_errors = state_action_values - expected_state_action_values
    losses_per_sample = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
    loss = (weights_v * losses_per_sample).mean()

    return loss, td_errors.detach().cpu().numpy()
