import torch
import torch.nn.functional  as F
import random


#used to create evironment representing the whole game.
def make_env():
    return#write it later


#gumbel sigmoid sampling related methods
"""
def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的multi hot'''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])
"""

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """从Gumbel(0,1)分布中采样"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid_sample(logits, temperature, device='cpu'):
    """从Gumbel-Sigmoid分布中采样"""
    gumbels = sample_gumbel(logits.shape, device=device)
    y = logits + gumbels
    return torch.sigmoid(y / temperature)  # 使用sigmoid替代softmax

def gumbel_sigmoid(logits, temperature=1.0, hard=False, device='cpu'):
    """
    Gumbel-Sigmoid采样（支持multi-hot）
    
    参数:
        logits: [batch_size, action_dim] 未归一化的动作分数
        temperature: 温度参数（τ），控制采样尖锐程度
        hard: 是否返回离散化的hard multi-hot
        device: 设备（'cpu'或'cuda'）
    
    返回:
        - 若 hard=True: 近似multi-hot的张量（梯度仍可导）
        - 若 hard=False: 连续概率张量
    """
    y = gumbel_sigmoid_sample(logits, temperature, device)
    
    if hard:
        # 离散化为hard multi-hot（0或1），但保持梯度
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y  # 梯度绕过离散化操作
    return y
    """
    训练时：必须用 hard=False(连续概率)，因为：
    梯度需要从 Critic 的 Q(s,a) 反向传播到策略网络。
    离散的 0/1 会阻断梯度，导致策略无法优化。
    ​执行时：必须用 hard=True(离散动作)，因为：
    环境需要明确的动作指令。
    ​类比：教练与运动员

    ​训练(hard=False)​: 教练用录像（连续反馈）调整运动员的动作细节（梯度下降）。
    ​比赛(ard=True)​: 运动员必须做出明确的离散动作（如“挥拍”或“不挥拍”）。
    """


#Net work structure
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#the algorithm part of DDPG
class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim, actor_lr, critic_lr, device):

        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_sigmoid(action)
        else:
            action = gumbel_sigmoid(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)



#MADDPG Class
class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.env = env

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(self.env.agents))
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        #the obs here are exactlly the state in our scenario: observation is assuming each agent can't
        #have the entire game state but our's can.
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        #all target act has N tensors. where N is the number of agent.
        #Each gubel_sigmid will simple out a tensor that the size is [batch_size, action_dimention]
        all_target_act = [
            gumbel_sigmoid(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        #target_critic_input 拼接后的形状为[batch_size, action_dim+state_dim]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)#中心化critic
        target_critic_value = rew[i_agent].view( -1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_sigmoid(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(gumbel_sigmoid(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)