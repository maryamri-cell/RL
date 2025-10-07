import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym 
from gymnasium import spaces 
import pickle 
import time
import os

# --- PARTIE I: ENVIRONNEMENT (GridWorld) ---

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


# --- PARTIE II: AGENT (SARSA Linéaire) ---

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
    
    def learn(self, state, action, reward, next_state, next_action):
        
        q_s_a = self.estimate_q(state, action)
        
        if next_state in self.env.goals or next_action is None:
             q_s_prime_a_prime = 0.0
        else:
             q_s_prime_a_prime = self.estimate_q(next_state, next_action)
        
        td_target = reward + self.gamma * q_s_prime_a_prime
        td_error = td_target - q_s_a
        
        phi_s_a = self.feature_extractor(state, action)
        self.theta += self.alpha * td_error * phi_s_a

    def decay_epsilon(self, decay_rate=0.995): 
        self.epsilon = max(0.05, self.epsilon * decay_rate) 

    def save_weights(self, filename="sarsa_linear_weights.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.theta, f)
        print(f"Poids theta sauvegardés dans {filename}")


# --- PARTIE III: FONCTIONS D'EXÉCUTION ET DE VISUALISATION ---

def train_agent(env, agent, num_episodes=500):
    print(f"\n--- Démarrage de l'entraînement SARSA Linéaire sur {num_episodes} épisodes ---")
    reward_history = []
    theta_history = []
    log_interval = 20
    
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
            
            agent.learn(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            episode_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(episode_reward)
        
        if (episode + 1) % log_interval == 0 or episode == num_episodes - 1:
            theta_history.append(agent.theta.copy())

        if (episode + 1) % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"Épisode {episode + 1}/{num_episodes}. Récompense: {episode_reward:.1f}. Epsilon: {agent.epsilon:.4f}.")
            
    print("--- Entraînement SARSA Linéaire Terminé ---")
    return reward_history, theta_history

def run_episode(env, agent, max_steps=50):
    
    observation, info = env.reset()
    env.total_reward = 0 
    
    done = False
    truncated = False
    step_count = 0
    
    env.render() 
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    while not done and not truncated:
        step_count += 1
        
        if step_count >= max_steps:
             truncated = True
        
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.get_action(observation)
        agent.epsilon = old_epsilon 

        observation, reward, done, truncated_step, info = env.step(action) 
        truncated = truncated or truncated_step

        print(f"Pas {step_count}: Action: {action_names[action]}, Pos: {observation}, Reward: {reward:.1f}")
        
        env.render() 

    print("\n================================")
    print(f"Épisode Terminé en {step_count} pas.")
    print(f"Récompense Totale: {env.total_reward:.1f}")
    print("================================\n")
    env.close() 

def plot_convergence(reward_history, title="Courbe de Convergence SARSA Linéaire"):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Récompense Totale par Épisode', color='blue', alpha=0.5)
    
    window = max(1, len(reward_history) // 10) 
    rolling_mean = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(reward_history)), rolling_mean, label=f'Moyenne Glissante (Fenêtre: {window})', color='orange', linewidth=2)
    
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense Totale Cumulée")
    plt.grid(True)
    plt.legend()

def plot_sensitivity(theta_history, log_interval, num_features, title="Sensibilité des Poids Theta (Convergence des Paramètres)"):
    if not theta_history:
        print("L'historique des poids theta est vide. Impossible de tracer la sensibilité.")
        return
        
    theta_matrix = np.array(theta_history)
    episodes = np.arange(log_interval, len(theta_history) * log_interval + log_interval, log_interval)
    
    # Ajuster la taille des épisodes si le dernier point a été ajouté
    if len(episodes) > len(theta_matrix):
        episodes = episodes[:len(theta_matrix)]
    elif len(episodes) < len(theta_matrix):
        episodes = np.append(episodes, episodes[-1] + log_interval) 
        episodes[-1] = (len(theta_matrix)-1) * log_interval + 1 
    episodes[-1] = max(episodes) # S'assurer que le dernier point est bien le dernier épisode

    plt.figure(figsize=(10, 5))
    
    for i in range(num_features):
        plt.plot(episodes, theta_matrix[:, i], alpha=0.7, 
                 label=f'$\\theta_{i}$')
    
    plt.title(title)
    plt.xlabel(f"Épisode (Log: {log_interval} ép.)")
    plt.ylabel("Valeur du Poids $\\theta_i$")
    plt.grid(True)
    plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small')
    plt.tight_layout(rect=[0, 0.15, 1, 1])


# --- PARTIE IV: EXÉCUTION PRINCIPALE (CORRIGÉE) ---

if __name__ == "__main__":
    
    # 1. Utiliser plt.ion() pour l'affichage du rendu de l'environnement pendant l'entraînement/test
    plt.ion() 
    
    GRID_SIZE = 7
    FIXED_GOALS = [(6, 6)]
    FIXED_OBSTACLES = [(2, 2), (2, 3), (4, 4), (4, 5)]
    NUM_TRAIN_EPISODES = 10000
    LOG_INTERVAL = 20

    print("\n--- DÉBUT DE L'ENTRAÎNEMENT SARSA LINÉAIRE ---")
    
    env_sarsa_train = StochasticGymnasiumGridWorld(
        grid_size=GRID_SIZE, 
        goals=FIXED_GOALS, 
        obstacles=FIXED_OBSTACLES, 
        render_mode=None,
        action_noise=0.1
    )
    
    agent_sarsa = LinearSARSAAgent(env_sarsa_train, alpha=0.005, gamma=0.99, epsilon=1.0) 
    
    reward_history, theta_history = train_agent(env_sarsa_train, agent_sarsa, num_episodes=NUM_TRAIN_EPISODES)
    env_sarsa_train.close()
    
    agent_sarsa.save_weights()
    
    # 1. Tracé de la courbe de convergence (Récompenses)
    plot_convergence(reward_history, title="1. Courbe de Convergence (Récompense Totale par Épisode)")
    
    # 2. Tracé de la courbe de sensibilité (Poids Theta)
    plot_sensitivity(theta_history, LOG_INTERVAL, agent_sarsa.num_features, 
                     title="2. Courbe de Sensibilité (Évolution des Poids $\\theta$)")
    
    print("\n--- TEST: Agent SARSA Linéaire (Exploitation) avec Rendu ---")
    env_run_sarsa = StochasticGymnasiumGridWorld(
        grid_size=GRID_SIZE, 
        goals=FIXED_GOALS, 
        obstacles=FIXED_OBSTACLES, 
        render_mode="human",
        action_noise=0.0
    )
    
    run_episode(env_run_sarsa, agent_sarsa, max_steps=40)
    plt.ioff() # Désactiver le mode interactif
    plt.show(block=True) # Afficher et bloquer le script
    
    print("\n--- FIN DU PROGRAMME ---")