import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import random
import gymnasium as gym
from torch.optim.lr_scheduler import StepLR
import os  # Import the 'os' module

# --- Setup ---
seed = 543  # Do not change this

def fix(env, seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Use gymnasium's make function with the new version and rendering
env = gym.make('LunarLander-v3', render_mode='rgb_array')  # Changed to v3
fix(env, seed)

# --- Use GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 128),  # Increased hidden layer size
            nn.ReLU(),        # Changed to ReLU
            nn.Linear(128, 128), # Increased hidden layer size
            nn.ReLU()         # Changed to ReLU
        )

        self.actor = nn.Linear(128, 4)  # Output layer for actions
        self.critic = nn.Linear(128, 1) # Output layer for value

        self.optimizer = optim.Adam(self.parameters(), lr=0.0003) # Changed optimizer and learning rate
        self.values = [] # values are stored per batch


    def forward(self, state):
        hid = self.fc(state)
        self.values.append(self.critic(hid).squeeze(-1))  # Store critic value
        return F.softmax(self.actor(hid), dim=-1)

    def learn(self, log_probs, rewards):
        values = torch.stack(self.values)
        loss = (-log_probs * (rewards - values.detach())).sum() + F.smooth_l1_loss(values, rewards) # Critic loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.values = [] # Clear values after update


    def sample(self, state):
        action_prob = self(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

agent = ActorCritic().to(device)  # Move agent to GPU

agent.train()
EPISODE_PER_BATCH = 5
NUM_BATCH = 100  # Increased the number of batches significantly
rate = 0.99

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm.tqdm(range(NUM_BATCH))  # Use tqdm.tqdm for notebook compatibility
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    for episode in range(EPISODE_PER_BATCH):
        state, _ = env.reset(seed=seed)  # Reset the environment and set seed
        state = torch.FloatTensor(state).to(device)

        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            seq_rewards.append(reward)

            next_state = torch.FloatTensor(next_state).to(device)
            state = next_state
            total_reward += reward
            total_step += 1

            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                for i in range(2, len(seq_rewards)+1):
                    seq_rewards[-i] += rate * (seq_rewards[-i+1])
                rewards += seq_rewards
                break

    # record training process
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    rewards = np.array(rewards)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards).to(device))



plt.plot(avg_total_rewards)
plt.title("Total Rewards of Actor Critic")
plt.show()

plt.plot(avg_final_rewards)
plt.title("Final Rewards of Actor Critic")
plt.show()

# --- Save Model ---
model_save_path = "actor_critic_model.pth"  # Define path to save the model
torch.save(agent.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# --- Testing Phase ---
fix(env, seed)
agent.eval()  # set the network into evaluation mode
NUM_OF_TEST = 5 # Do not revise this !!!
test_total_reward = []
action_list = []


# Create a figure and axes for the display *outside* the loop.  Crucially, make it non-blocking.
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()


for i in range(NUM_OF_TEST):
    actions = []
    state, _ = env.reset(seed=seed)
    state = torch.FloatTensor(state).to(device)

    img = ax.imshow(env.render())
    ax.set_title(f"Test Run {i+1}")

    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            action, _ = agent.sample(state)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.FloatTensor(next_state).to(device)
        total_reward += reward

        # Update the image and *redraw* the figure.
        img.set_data(env.render())
        fig.canvas.draw()
        fig.canvas.flush_events()  # Process events to ensure the window updates.
        plt.pause(0.01)  # Add a small pause.

    print(f"Test Run {i+1}: Total Reward = {total_reward}")
    test_total_reward.append(total_reward)
    action_list.append(actions)

print(f"Test Total Rewards: {test_total_reward}")
print(f"Average Test Reward: {np.mean(test_total_reward)}")

plt.ioff()  # Turn off interactive mode
plt.close(fig) # Close figure

