import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym 
from gymnasium import spaces 
import pickle 
class GymnasiumGridWorld(gym.Env): 
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, goals=[(4, 4)], obstacles=[(2, 2)], render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.goals = set(goals) 
        self.obstacles = set(obstacles) 
        self.start_pos = (0, 0) 
        
        
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
        """Configure les éléments statiques de la grille pour le rendu."""
        
        self.grid_matrix = np.zeros((self.grid_size, self.grid_size))
        
        for r, c in self.goals:
            self.grid_matrix[r, c] = 1 
        for r, c in self.obstacles:
            self.grid_matrix[r, c] = 2 
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
            self.ax.text(c, r, 'GOAL', ha='center', va='center', color='white' if r>0 or c>0 else 'black', fontsize=10, fontweight='bold')
        for r, c in self.obstacles:
            self.ax.text(c, r, 'OBS', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            
        self.agent_marker, = self.ax.plot([], [], marker='o', markersize=20, 
                                          color='green', linestyle='', label='Agent')


    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        self.state = self.start_pos # (0, 0)
        self.terminated = False
        self.truncated = False 
        self.total_reward = 0.0

        observation = self.state
        info = {} 
        
        return observation, info

    def get_transition_prob(self, state, action):
        """
        [MODÈLE] Retourne la dynamique pour les algorithmes Model-Based (PI/VI) et 
        est utilisée par step() pour les Model-Free.
        """
        row, col = state
        new_row, new_col = row, col 
        
        if action == 0: new_row = max(0, row - 1) # UP
        elif action == 1: new_row = min(self.grid_size - 1, row + 1) # DOWN
        elif action == 2: new_col = max(0, col - 1) # LEFT
        elif action == 3: new_col = min(self.grid_size - 1, col + 1) # RIGHT
        
        next_state_candidate = (new_row, new_col)
        if next_state_candidate in self.obstacles:
            next_state = state 
            reward = -5         
            is_terminated = False
        else:
            next_state = next_state_candidate
            reward = -1 
            is_terminated = False

            if next_state in self.goals:
                reward = 10
                is_terminated = True

        return next_state, reward, is_terminated


    def step(self, action):
        """Exécute l'action et retourne les 5 éléments requis par l'API Gymnasium."""
        if self.terminated or self.truncated:
            return self.state, 0, self.terminated, self.truncated, {}
        new_state, reward, terminated = self.get_transition_prob(self.state, action)
        
        self.state = new_state
        self.terminated = terminated
        return self.state, reward, self.terminated, self.truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            self.agent_marker.set_data([self.state[1]], [self.state[0]]) 
            self.ax.set_title(f"Pos: {self.state} | R: {self.total_reward:.1f} | Term: {self.terminated}", fontsize=10)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1/self.metadata["render_fps"])
            
        elif self.render_mode is not None:
            pass 

    def close(self):
        """Ferme toutes les ressources de rendu."""
        if hasattr(self, 'fig') and self.fig:
             plt.close(self.fig)



class QLearningAgent:
    """Agent Q-Learning (Model-Free) - Adapté pour les obstacles."""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.Q = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    def get_action(self, state):
        """Politique Epsilon-Greedy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            r, c = state
            return np.argmax(self.Q[r, c, :])
    
    def learn(self, state, action, reward, next_state):
        """Mise à jour de la table Q."""
        r, c = state
        r_prime, c_prime = next_state
        
        # Q-Learning Update Rule
        max_q_next = np.max(self.Q[r_prime, c_prime, :])
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.Q[r, c, action]
        
        self.Q[r, c, action] += self.alpha * td_error

    def decay_epsilon(self, decay_rate=0.999):
        """Diminution de l'exploration."""
        self.epsilon = max(0.05, self.epsilon * decay_rate) 
        
    def save_q_table(self, filename="q_table.pkl"):
        """Sauvegarde la table Q dans un fichier binaire."""
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Table Q sauvegardée dans {filename}")
        
    def load_q_table(self, filename="q_table.pkl"):
        """Charge la table Q depuis un fichier binaire."""
        try:
            with open(filename, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Table Q chargée depuis {filename}")
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. La table Q reste à zéro.")


def train_agent(env, agent, num_episodes=50):
    print(f"\n--- Démarrage de l'entraînement Q-Learning sur {num_episodes} épisodes ---")
    reward_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        while not done and not truncated:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        reward_history.append(episode_reward)
        
        if (episode + 1) % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"Épisode {episode + 1}/{num_episodes}. Récompense: {episode_reward:.1f}. Epsilon: {agent.epsilon:.4f}")

    print("--- Entraînement Q-Learning Terminé ---")
    return reward_history

def run_episode(env, agent, max_steps=50):
    """Exécute un épisode en mode pure exploitation et avec rendu."""
    
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
        
        action = np.argmax(agent.Q[observation[0], observation[1], :]) if isinstance(agent, QLearningAgent) else agent.get_action(observation)
        
        observation, reward, done, truncated_step, info = env.step(action) 
        truncated = truncated or truncated_step
        env.total_reward += reward

        print(f"Pas {step_count}: Action: {action_names[action]}, Pos: {observation}, Reward: {reward}")
        
        env.render() 

    print("\n================================")
    print(f"Épisode Terminé en {step_count} pas.")
    print(f"Statut: {'GOAL' if done else ('TRUNCATED' if truncated else 'Unknown')}")
    print(f"Récompense Totale: {env.total_reward}")
    print("================================\n")
    env.close() 


def plot_convergence(reward_history, title="Courbe de Convergence Q-Learning"):
    """Trace l'historique des récompenses cumulées."""
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Récompense Totale par Épisode', color='blue')
    
    window = max(1, len(reward_history) // 10) 
    rolling_mean = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(reward_history)), rolling_mean, label=f'Moyenne Glissante (Fenêtre: {window})', color='orange', linewidth=2)
    
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense Totale Cumulée")
    plt.grid(True)
    plt.legend()
    plt.show()

def run_sensitivity_analysis(grid_sizes, num_episodes=50, num_runs=5):
    all_rewards = {}
    
    for size in grid_sizes:
        print(f"\n--- Test de la taille de grille {size}x{size} ---")
        size_rewards = []
        goals = [(size - 1, size - 1)] 
        obstacles = [(size // 2, size // 2)]
        
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}...")
            env = GymnasiumGridWorld(grid_size=size, goals=goals, obstacles=obstacles, render_mode=None)
            agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=1.0)
            history = train_agent(env, agent, num_episodes=num_episodes)
            size_rewards.append(history)
            env.close()
            
        all_rewards[size] = size_rewards
        
    plt.figure(figsize=(12, 6))
    
    for size, rewards_list in all_rewards.items():
        
        mean_rewards = np.mean(rewards_list, axis=0)
        std_rewards = np.std(rewards_list, axis=0)
        
       
        plt.plot(mean_rewards, label=f'Grille {size}x{size} (Moyenne)')
        
        
        plt.fill_between(range(num_episodes), 
                         mean_rewards - std_rewards, 
                         mean_rewards + std_rewards, 
                         alpha=0.15, label=f'Grille {size}x{size} (Variance)')

    plt.title(f"Sensibilité du Q-Learning à la Taille de Grille (Moyenne et Variance sur {num_runs} Runs)")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense Totale Moyenne")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plt.ion() 
    
    
    GRID_SIZE = 10
    FIXED_GOALS = [(6, 6),(3,3)]
    FIXED_OBSTACLES = [(2, 2), (4, 4), (5, 1)]
    NUM_TRAIN_EPISODES = 20000
    
    
    print("\n--- DÉBUT DE L'ENTRAÎNEMENT Q-LEARNING ---")
    env_q_train = GymnasiumGridWorld(grid_size=GRID_SIZE, goals=FIXED_GOALS, 
                                     obstacles=FIXED_OBSTACLES, render_mode=None)
    agent_q = QLearningAgent(env_q_train, alpha=0.1, gamma=0.9, epsilon=1.0)
    
    
    reward_history = train_agent(env_q_train, agent_q, num_episodes=NUM_TRAIN_EPISODES)
    env_q_train.close()
    
    agent_q.save_q_table(filename="q_table_fixed_7x7.pkl")
    
    plot_convergence(reward_history, title=f"Convergence Q-Learning ({GRID_SIZE}x{GRID_SIZE}, {NUM_TRAIN_EPISODES} Épisodes)")
    
    print("\n--- TEST: Agent Q-Learning (Exploitation) avec Rendu ---")
    env_run_q = GymnasiumGridWorld(grid_size=GRID_SIZE, goals=FIXED_GOALS, 
                                     obstacles=FIXED_OBSTACLES, render_mode="human")
    agent_q.epsilon = 0.0 
    run_episode(env_run_q, agent_q, max_steps=2000000)
    
    plt.ioff() 
    run_sensitivity_analysis(grid_sizes=[5, 7, 10], num_episodes=100000000, num_runs=5)
    plt.ion() 
    print("\n--- FIN DU PROGRAMME ---")
    plt.ioff()