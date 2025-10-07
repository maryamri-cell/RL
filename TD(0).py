import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym 
from gymnasium import spaces 
import pickle 
import time
from collections import defaultdict


class StochasticGymnasiumGridWorld(gym.Env): 
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, goals=[(3, 3)], obstacles=[(2, 2)], render_mode=None, action_noise=0.2):
        super().__init__()
        
        self.grid_size = grid_size
        self.goals = set(goals) 
        self.obstacles = set(obstacles) 
        self.start_pos = (0, 0)
        self.action_noise = action_noise

        if not all(0 <= r < grid_size and 0 <= c < grid_size for r, c in self.goals):
             raise ValueError("Les coordonnées des goals doivent être dans la grille.")
        if not all(0 <= r < grid_size and 0 <= c < grid_size for r, c in self.obstacles):
             raise ValueError("Les coordonnées des obstacles doivent être dans la grille.")
        if self.start_pos in self.goals or self.start_pos in self.obstacles:
             raise ValueError("La position de départ ne doit être ni un goal ni un obstacle.")
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size)
        ))
        
        self.render_mode = render_mode
        self.total_reward = 0.0 
        
        if self.render_mode == "human":
            plt.ion() 
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self._setup_render()

    def _setup_render(self):
        if hasattr(self, 'agent_marker'):
            self.agent_marker.remove()
        
        self.grid_matrix = np.zeros((self.grid_size, self.grid_size))
        for r, c in self.goals: self.grid_matrix[r, c] = 1 
        for r, c in self.obstacles: self.grid_matrix[r, c] = 2 
            
        colors = ['white', 'red', 'black'] 
        self.cmap = mcolors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        
        self.im = self.ax.imshow(self.grid_matrix, cmap=self.cmap, norm=self.norm)
        
        self.ax.set_xticks(np.arange(self.grid_size + 1) - 0.5, minor=False)
        self.ax.set_yticks(np.arange(self.grid_size + 1) - 0.5, minor=False)
        self.ax.tick_params(axis='both', which='both', length=0)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(which='major', color='gray', linestyle='-', linewidth=1.5)
        
        for r, c in self.goals:
             color_text = 'black' if r == 0 and c == 0 else 'white'
             self.ax.text(c, r, 'GOAL', ha='center', va='center', color=color_text, fontsize=10, fontweight='bold')
        for r, c in self.obstacles:
             self.ax.text(c, r, 'OBS', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            
        self.agent_marker, = self.ax.plot([], [], marker='o', markersize=20, 
                                             color='green', linestyle='', label='Agent')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos
        self.terminated = False
        self.truncated = False 
        self.total_reward = 0.0
        observation = self.state
        info = {} 
        return observation, info

    def get_actual_move(self, action):
        if random.random() < self.action_noise:
            deviations = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}
            return random.choice(deviations.get(action, [0, 1, 2, 3]))
        return action

    def step(self, action):
        if self.terminated or self.truncated:
            return self.state, 0, self.terminated, self.truncated, {}
        
        actual_action = self.get_actual_move(action)

        row, col = self.state
        new_row, new_col = row, col 
        
        if actual_action == 0: new_row = max(0, row - 1)
        elif actual_action == 1: new_row = min(self.grid_size - 1, row + 1)
        elif actual_action == 2: new_col = max(0, col - 1)
        elif actual_action == 3: new_col = min(self.grid_size - 1, col + 1)
        
        next_state_candidate = (new_row, new_col)
        
        if next_state_candidate in self.obstacles:
            next_state = self.state
            reward = -5
            terminated = False
        else:
            next_state = next_state_candidate
            reward = -0.5
            terminated = False
            if next_state in self.goals:
                reward = 10
                terminated = True

        self.state = next_state
        self.terminated = terminated
        
        if self.total_reward < -self.grid_size * 5: 
            self.truncated = True
        
        self.total_reward += reward

        return self.state, reward, self.terminated, self.truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            self.agent_marker.set_data([self.state[1]], [self.state[0]]) 
            self.ax.set_title(f"Pos: {self.state} | R: {self.total_reward:.1f} | Term: {self.terminated}", fontsize=10)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1/self.metadata["render_fps"])

    def close(self):
        if hasattr(self, 'fig') and self.fig:
             plt.close(self.fig)


# 1. Q-LEARNING (Tabulaire)
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon = max(0.05, self.epsilon * decay_rate)


# 2. SARSA (Tabulaire)
class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, next_action, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state][next_action]
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon = max(0.05, self.epsilon * decay_rate)


# 3. Expected SARSA (Tabulaire)
class ExpectedSARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            # Espérance sur toutes les actions possibles
            q_values = self.q_table[next_state]
            best_action = np.argmax(q_values)
            
            expected_q = 0
            for a in range(self.env.action_space.n):
                if a == best_action:
                    prob = 1 - self.epsilon + self.epsilon / self.env.action_space.n
                else:
                    prob = self.epsilon / self.env.action_space.n
                expected_q += prob * q_values[a]
            
            td_target = reward + self.gamma * expected_q
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon = max(0.05, self.epsilon * decay_rate)


# 4. Double Q-Learning (Tabulaire)
class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_a = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q_table_b = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Moyenne des deux tables Q
            q_avg = (self.q_table_a[state] + self.q_table_b[state]) / 2
            return np.argmax(q_avg)
    
    def learn(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            # Mise à jour aléatoire d'une des deux tables
            if random.random() < 0.5:
                best_action = np.argmax(self.q_table_a[next_state])
                td_target = reward + self.gamma * self.q_table_b[next_state][best_action]
                td_error = td_target - self.q_table_a[state][action]
                self.q_table_a[state][action] += self.alpha * td_error
            else:
                best_action = np.argmax(self.q_table_b[next_state])
                td_target = reward + self.gamma * self.q_table_a[next_state][best_action]
                td_error = td_target - self.q_table_b[state][action]
                self.q_table_b[state][action] += self.alpha * td_error
            return
        
        # Si terminé, mise à jour des deux tables
        td_error_a = reward - self.q_table_a[state][action]
        td_error_b = reward - self.q_table_b[state][action]
        self.q_table_a[state][action] += self.alpha * td_error_a
        self.q_table_b[state][action] += self.alpha * td_error_b
    
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon = max(0.05, self.epsilon * decay_rate)


# 5. SARSA Linéaire (Approximation de fonction)
class LinearSARSAAgent:
    def __init__(self, env, alpha=0.005, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        
        self.num_base_features = 5
        self.num_actions = env.action_space.n 
        self.num_features = self.num_base_features * self.num_actions
        
        self.theta = np.zeros(self.num_features) 

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def feature_extractor(self, state, action):
        if action < 0 or action >= self.num_actions:
            return np.zeros(self.num_features)

        r, c = state
        
        if self.env.goals:
            dist_goal = min(self._manhattan_distance(state, g) for g in self.env.goals)
        else:
            dist_goal = self.env.grid_size * 2
            
        if self.env.obstacles:
            dist_obs_min = min(self._manhattan_distance(state, o) for o in self.env.obstacles)
        else:
            dist_obs_min = self.env.grid_size * 2
        
        phi_base = np.array([r, c, dist_goal, dist_obs_min, 1.0])
        
        phi_s_a = np.zeros(self.num_features)
        start_index = action * self.num_base_features
        phi_s_a[start_index : start_index + self.num_base_features] = phi_base
        
        return phi_s_a

    def estimate_q(self, state, action):
        if state in self.env.goals:
            return 0.0
        phi_s_a = self.feature_extractor(state, action)
        return np.dot(self.theta, phi_s_a)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.estimate_q(state, action) for action in range(self.num_actions)]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, next_action, done):
        q_s_a = self.estimate_q(state, action)
        
        if done or next_action is None:
             q_s_prime_a_prime = 0.0
        else:
             q_s_prime_a_prime = self.estimate_q(next_state, next_action)
        
        td_target = reward + self.gamma * q_s_prime_a_prime
        td_error = td_target - q_s_a
        
        phi_s_a = self.feature_extractor(state, action)
        self.theta += self.alpha * td_error * phi_s_a

    def decay_epsilon(self, decay_rate=0.995): 
        self.epsilon = max(0.05, self.epsilon * decay_rate)


# 6. Q-Learning Linéaire (Approximation de fonction)
class LinearQLearningAgent:
    def __init__(self, env, alpha=0.005, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        
        self.num_base_features = 5
        self.num_actions = env.action_space.n 
        self.num_features = self.num_base_features * self.num_actions
        
        self.theta = np.zeros(self.num_features) 

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def feature_extractor(self, state, action):
        if action < 0 or action >= self.num_actions:
            return np.zeros(self.num_features)

        r, c = state
        
        if self.env.goals:
            dist_goal = min(self._manhattan_distance(state, g) for g in self.env.goals)
        else:
            dist_goal = self.env.grid_size * 2
            
        if self.env.obstacles:
            dist_obs_min = min(self._manhattan_distance(state, o) for o in self.env.obstacles)
        else:
            dist_obs_min = self.env.grid_size * 2
        
        phi_base = np.array([r, c, dist_goal, dist_obs_min, 1.0])
        
        phi_s_a = np.zeros(self.num_features)
        start_index = action * self.num_base_features
        phi_s_a[start_index : start_index + self.num_base_features] = phi_base
        
        return phi_s_a

    def estimate_q(self, state, action):
        if state in self.env.goals:
            return 0.0
        phi_s_a = self.feature_extractor(state, action)
        return np.dot(self.theta, phi_s_a)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.estimate_q(state, action) for action in range(self.num_actions)]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        q_s_a = self.estimate_q(state, action)
        
        if done:
             q_max_next = 0.0
        else:
             q_max_next = max([self.estimate_q(next_state, a) for a in range(self.num_actions)])
        
        td_target = reward + self.gamma * q_max_next
        td_error = td_target - q_s_a
        
        phi_s_a = self.feature_extractor(state, action)
        self.theta += self.alpha * td_error * phi_s_a

    def decay_epsilon(self, decay_rate=0.995): 
        self.epsilon = max(0.05, self.epsilon * decay_rate)



def train_qlearning(env, agent, num_episodes=500):
    reward_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        while not done and not truncated:
            action = agent.get_action(state)
            next_state, reward, done, truncated_step, _ = env.step(action)
            truncated = truncated or truncated_step
            
            agent.learn(state, action, reward, next_state, done or truncated)
            
            state = next_state
            episode_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(episode_reward)
            
    return reward_history


def train_sarsa(env, agent, num_episodes=500):
    reward_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        action = agent.get_action(state)
        
        while not done and not truncated:
            next_state, reward, done, truncated_step, _ = env.step(action)
            truncated = truncated or truncated_step
            
            if done or truncated:
                next_action = None
            else:
                next_action = agent.get_action(next_state)
            
            agent.learn(state, action, reward, next_state, next_action, done or truncated)
            
            state = next_state
            action = next_action
            episode_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(episode_reward)
            
    return reward_history


def plot_comparison(results_dict, title="Comparaison des Algorithmes TD(0)", window=50):
    """Compare les performances de plusieurs algorithmes"""
    plt.figure(figsize=(14, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (algo_name, rewards) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        
        # Courbe brute avec transparence
        plt.plot(rewards, alpha=0.2, color=color)
        
        # Moyenne glissante
        if len(rewards) >= window:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(rewards)), rolling_mean, 
                    label=algo_name, color=color, linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Épisode", fontsize=12)
    plt.ylabel("Récompense Totale", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()


def plot_convergence_speed(results_dict, threshold=-10):
    """Analyse la vitesse de convergence"""
    plt.figure(figsize=(10, 6))
    
    algo_names = []
    episodes_to_converge = []
    
    for algo_name, rewards in results_dict.items():
        # Trouver le premier épisode où la moyenne glissante dépasse le seuil
        window = 50
        if len(rewards) >= window:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            converged = np.where(rolling_mean >= threshold)[0]
            if len(converged) > 0:
                episodes_to_converge.append(converged[0] + window)
            else:
                episodes_to_converge.append(len(rewards))
        else:
            episodes_to_converge.append(len(rewards))
        
        algo_names.append(algo_name)
    
    bars = plt.bar(algo_names, episodes_to_converge, color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    plt.title(f"Vitesse de Convergence (Seuil: {threshold})", fontsize=14, fontweight='bold')
    plt.xlabel("Algorithme", fontsize=12)
    plt.ylabel("Épisodes pour Converger", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()


def plot_final_performance(results_dict, last_n=100):
    """Compare les performances finales"""
    plt.figure(figsize=(10, 6))
    
    algo_names = []
    final_rewards = []
    
    for algo_name, rewards in results_dict.items():
        algo_names.append(algo_name)
        final_rewards.append(np.mean(rewards[-last_n:]))
    
    bars = plt.bar(algo_names, final_rewards, color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    plt.title(f"Performance Finale (Moyenne sur les {last_n} derniers épisodes)", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Algorithme", fontsize=12)
    plt.ylabel("Récompense Moyenne", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()


def plot_variance_comparison(results_dict, window=50):
    """Compare la variance des performances"""
    plt.figure(figsize=(14, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (algo_name, rewards) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        
        if len(rewards) >= window:
            # Calcul de la variance glissante
            rolling_var = []
            for i in range(window, len(rewards)):
                rolling_var.append(np.var(rewards[i-window:i]))
            
            plt.plot(np.arange(window, len(rewards)), rolling_var, 
                    label=algo_name, color=color, linewidth=2)
    
    plt.title("Stabilité des Algorithmes (Variance Glissante)", fontsize=14, fontweight='bold')
    plt.xlabel("Épisode", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()


def run_episode_visual(env, agent, max_steps=50, is_sarsa=False):
    """Exécute un épisode avec visualisation"""
    observation, info = env.reset()
    env.total_reward = 0 
    
    done = False
    truncated = False
    step_count = 0
    
    env.render() 
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    if is_sarsa:
        action = agent.get_action(observation)

    while not done and not truncated:
        step_count += 1
        
        if step_count >= max_steps:
             truncated = True
        
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        if not is_sarsa:
            action = agent.get_action(observation)
        
        agent.epsilon = old_epsilon 

        next_observation, reward, done, truncated_step, info = env.step(action) 
        truncated = truncated or truncated_step

        print(f"Pas {step_count}: Action: {action_names[action]}, Pos: {next_observation}, Reward: {reward:.1f}")
        
        env.render()
        
        if is_sarsa and not done and not truncated:
            agent.epsilon = 0.0
            action = agent.get_action(next_observation)
            agent.epsilon = old_epsilon
        
        observation = next_observation

    print(f"\nÉpisode Terminé en {step_count} pas. Récompense: {env.total_reward:.1f}\n")
    env.close()



if __name__ == "__main__":
    
    GRID_SIZE = 7
    FIXED_GOALS = [(6, 6)]
    FIXED_OBSTACLES = [(2, 2), (2, 3), (4, 4), (4, 5)]
    NUM_EPISODES = 5000
    ACTION_NOISE = 0.1
    
    print("\n" + "="*60)
    print("COMPARAISON DES ALGORITHMES TD(0)")
    print("="*60 + "\n")
    
    results = {}
    
    # 1. Q-Learning Tabulaire
    print("Entraînement: Q-Learning (Tabulaire)...")
    env1 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent1 = QLearningAgent(env1, alpha=0.1, gamma=0.99, epsilon=1.0)
    results['Q-Learning'] = train_qlearning(env1, agent1, NUM_EPISODES)
    env1.close()
    
    # 2. SARSA Tabulaire
    print("Entraînement: SARSA (Tabulaire)...")
    env2 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent2 = SARSAAgent(env2, alpha=0.1, gamma=0.99, epsilon=1.0)
    results['SARSA'] = train_sarsa(env2, agent2, NUM_EPISODES)
    env2.close()
    
    # 3. Expected SARSA
    print("Entraînement: Expected SARSA...")
    env3 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent3 = ExpectedSARSAAgent(env3, alpha=0.1, gamma=0.99, epsilon=1.0)
    results['Expected SARSA'] = train_qlearning(env3, agent3, NUM_EPISODES)
    env3.close()
    
    # 4. Double Q-Learning
    print("Entraînement: Double Q-Learning...")
    env4 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent4 = DoubleQLearningAgent(env4, alpha=0.1, gamma=0.99, epsilon=1.0)
    results['Double Q-Learning'] = train_qlearning(env4, agent4, NUM_EPISODES)
    env4.close()
    
    # 5. SARSA Linéaire
    print("Entraînement: SARSA Linéaire...")
    env5 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent5 = LinearSARSAAgent(env5, alpha=0.005, gamma=0.99, epsilon=1.0)
    results['SARSA Linéaire'] = train_sarsa(env5, agent5, NUM_EPISODES)
    env5.close()
    
    # 6. Q-Learning Linéaire
    print("Entraînement: Q-Learning Linéaire...")
    env6 = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, None, ACTION_NOISE)
    agent6 = LinearQLearningAgent(env6, alpha=0.005, gamma=0.99, epsilon=1.0)
    results['Q-Learning Linéaire'] = train_qlearning(env6, agent6, NUM_EPISODES)
    env6.close()
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ - GÉNÉRATION DES GRAPHIQUES")
    print("="*60 + "\n")
    
    # Génération de toutes les visualisations
    plot_comparison(results, "1. Comparaison Globale des Algorithmes TD(0)")
    plot_convergence_speed(results, threshold=0)
    plot_final_performance(results, last_n=100)
    plot_variance_comparison(results, window=50)
    
    # Tableau récapitulatif
    print("\n" + "="*60)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("="*60)
    print(f"{'Algorithme':<25} {'Récomp. Finale':<18} {'Récomp. Max':<15}")
    print("-"*60)
    
    for algo_name, rewards in results.items():
        final_avg = np.mean(rewards[-100:])
        max_reward = np.max(rewards)
        print(f"{algo_name:<25} {final_avg:<18.2f} {max_reward:<15.2f}")
    
    print("="*60 + "\n")
    
    # Test visuel avec le meilleur agent
    print("Sélection du meilleur agent pour démonstration...")
    best_algo = max(results.items(), key=lambda x: np.mean(x[1][-100:]))
    print(f"Meilleur algorithme: {best_algo[0]}\n")
    
    # Choisir l'agent correspondant pour le test
    print("Démonstration visuelle avec Q-Learning (Tabulaire)...")
    env_demo = StochasticGymnasiumGridWorld(GRID_SIZE, FIXED_GOALS, FIXED_OBSTACLES, "human", 0.0)
    run_episode_visual(env_demo, agent1, max_steps=50, is_sarsa=False)
    
    plt.show(block=True)
    
    print("\n" + "="*60)
    print("ANALYSE TERMINÉE")
    print("="*60 + "\n")
    