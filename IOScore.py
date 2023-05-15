import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc = [torch.nn.Linear(state_dim + action_dim, hidden_dim[0]),]
        hidden_dim.append(1)
        for i in range(len(hidden_dim) - 1):
            self.fc.append(torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]))

    def forward(self, x, a):
        x = torch.cat([x, a], dim = 1)
        for i in range(len(self.fc)):
            x = F.relu(self.fc[i](x))
        return torch.sigmoid(self.fc2(x))

class IOScore:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.device = device

    def update(self, object_s, object_a, agent_s, agent_a, option):
        object_states = torch.tensor(object_s, dtype=torch.float).to(self.device)
        object_actions = torch.tensor(object_a).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a).to(self.device)

        object_prob = self.discriminator(object_states, object_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                object_prob, torch.zeros_like(object_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        IScore = -torch.log(agent_prob).detach().cpu().numpy() # The extent agent acts like the object (imitation)
        OScore = -torch.log(1 - agent_prob).detach().cpu().numpy() # The extent agent acts dislike the object (opposition)

        return IScore, OScore