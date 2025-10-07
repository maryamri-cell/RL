import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym 
from gymnasium import spaces 
import pickle 
import time
import os

# ====================================================================
# I. Environnement (Goal Aléatoire AVEC État 4D: (r_a, c_a, r_g, c_g))
# ====================================================================

class GymnasiumGridWorld(gym.Env): 
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=8, obstacles=[(2, 2), (4, 4), (5, 1)], render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        # Les obstacles doivent être ajustés pour rester dans les grilles de taille variable
        self.obstacles = set(o for o in obstacles if o[0] < grid_size and o[1] < grid_size)
        self.start_pos = (0, 0)
        
        self.all_states = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        self.valid_goal_locations = [s for s in self.all_states 
                                     if s not in self.obstacles and s != self.start_pos]
        
        if not self.valid_goal_locations:
             # Gérer le cas où la grille est trop petite ou l'obstacle recouvre le goal
             if grid_size > 1:
                self.valid_goal_locations = [(grid_size - 1, grid_size - 1)] 
             else:
                raise ValueError("Aucun endroit valide pour placer le goal.")

        self.current_goal = None 
        self.goals = set() 
        
        self.action_space = spaces.Discrete(4) 
        
        # L'espace d'observation DOIT refléter la nouvelle taille de grille
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size)
        ))
        
        self.render_mode = render_mode
        self.total_reward = 0.0 
        
        if self.render_mode == "human":
            plt.ion() 
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def _setup_render(self):
        if not self.current_goal: return
            
        if hasattr(self, 'im'):
            if hasattr(self, 'texts'):
                for t in self.texts: t.remove()
            self.im.remove()

        self.grid_matrix = np.zeros((self.grid_size, self.grid_size))
        
        r_g, c_g = self.current_goal
        self.grid_matrix[r_g, c_g] = 1 
        
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
        
        self.texts = []
        self.texts.append(self.ax.text(c_g, r_g, 'GOAL', ha='center', va='center', color='white', fontsize=10, fontweight='bold'))
        for r, c in self.obstacles:
            self.texts.append(self.ax.text(c, r, 'OBS', ha='center', va='center', color='white', fontsize=10, fontweight='bold'))
            
        if not hasattr(self, 'agent_marker'):
            self.agent_marker, = self.ax.plot([], [], marker='o', markersize=20, 
                                             color='green', linestyle='', label='Agent')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_goal = random.choice(self.valid_goal_locations)
        self.goals = {self.current_goal} 
        
        r_a, c_a = self.start_pos
        r_g, c_g = self.current_goal
        self.state = (r_a, c_a, r_g, c_g) 
        
        self.terminated = False
        self.truncated = False 
        self.total_reward = 0.0

        if self.render_mode == "human":
            self._setup_render()

        observation = self.state
        info = {} 
        return observation, info

    def get_transition_prob(self, state_4d, action):
        r_a, c_a = state_4d[0], state_4d[1]
        new_r_a, new_c_a = r_a, c_a 
        
        if action == 0: new_r_a = max(0, r_a - 1) 
        elif action == 1: new_r_a = min(self.grid_size - 1, r_a + 1)
        elif action == 2: new_c_a = max(0, c_a - 1)
        elif action == 3: new_c_a = min(self.grid_size - 1, c_a + 1)
        
        next_pos_candidate = (new_r_a, new_c_a)
        
        reward = -1 
        is_terminated = False
        
        if next_pos_candidate in self.obstacles:
            next_agent_pos = (r_a, c_a)
            reward = -5
        else:
            next_agent_pos = next_pos_candidate
            if next_agent_pos in self.goals:
                reward = 10
                is_terminated = True

        next_state = next_agent_pos + self.current_goal 
        
        return next_state, reward, is_terminated

    def step(self, action):
        if self.terminated or self.truncated:
            return self.state, 0, self.terminated, self.truncated, {}
        
        new_state, reward, terminated = self.get_transition_prob(self.state, action)
        
        self.state = new_state
        self.terminated = terminated
        self.total_reward += reward
        
        return self.state, reward, self.terminated, self.truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            r_a, c_a, r_g, c_g = self.state
            
            self.agent_marker.set_data([c_a], [r_a]) 
            current_goal_str = str((r_g, c_g))
            self.ax.set_title(f"Pos: ({r_a},{c_a}) | Goal: {current_goal_str} | R: {self.total_reward:.1f} | Term: {self.terminated}", fontsize=10)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1/self.metadata["render_fps"])

    def close(self):
        if hasattr(self, 'fig') and self.fig:
             plt.close(self.fig)

# ====================================================================
# II. Agent Q-Learning (Adapté à l'État 4D)
# ====================================================================

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        
        s = env.grid_size
        # Utiliser un dictionnaire pour stocker Q-table pour supporter différentes tailles de grille
        self.Q = {}

    # La méthode get_action et learn doivent gérer l'accès à Q via des clés de tuple
    def _get_q(self, state, action):
        """Accède à la Q-value, initialise à 0 si elle n'existe pas (nécessaire avec dict)."""
        key = state + (action,)
        return self.Q.get(key, 0.0)

    def _set_q(self, state, action, value):
        """Met à jour la Q-value (nécessaire avec dict)."""
        key = state + (action,)
        self.Q[key] = value

    def get_action(self, state):
        """Politique Epsilon-Greedy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self._get_q(state, a) for a in range(self.env.action_space.n)]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state):
        """Mise à jour de la table Q."""
        
        q_sa = self._get_q(state, action)
        
        # Trouver max Q pour next_state
        q_next_values = [self._get_q(next_state, a) for a in range(self.env.action_space.n)]
        max_q_next = np.max(q_next_values)
        
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - q_sa
        
        new_q_sa = q_sa + self.alpha * td_error
        self._set_q(state, action, new_q_sa)

    def decay_epsilon(self, decay_rate=0.99995):
        self.epsilon = max(0.01, self.epsilon * decay_rate) 

# ====================================================================
# III. Fonctions d'Entraînement et d'Analyse (MODIFIÉES)
# ====================================================================

def train_agent(env, agent, num_episodes=10000):
    print(f"\n--- Entraînement Q-Learning sur {num_episodes} épisodes (Grille {env.grid_size}x{env.grid_size}) ---")
    reward_history = []
    q_value_history = []
    log_interval = 100
    
    # État fixe à suivre (Agent en (0,0), Goal en (max, max))
    r_g_fixed, c_g_fixed = env.grid_size - 1, env.grid_size - 1
    state_to_track = (0, 0, r_g_fixed, c_g_fixed)
    action_to_track = 1 # Action: DOWN

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
        
        if (episode + 1) % log_interval == 0 or episode == num_episodes - 1:
            q_val = agent._get_q(state_to_track, action_to_track)
            q_value_history.append(q_val)


        if (episode + 1) % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"Épisode {episode + 1}/{num_episodes}. Récompense: {episode_reward:.1f}. Epsilon: {agent.epsilon:.5f}.")

    print(f"--- Entraînement Q-Learning (Grille {env.grid_size}x{env.grid_size}) Terminé ---")
    return reward_history, q_value_history, log_interval

# --- Sensibilité au nombre d'épisodes ---
def plot_convergence(reward_history, title="1. Courbe de Convergence (Récompense Totale par Épisode)"):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Récompense Totale par Épisode', color='blue', alpha=0.5)
    
    window = max(1, len(reward_history) // 20) 
    rolling_mean = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(reward_history)), rolling_mean, label=f'Moyenne Glissante (Fenêtre: {window})', color='orange', linewidth=2)
    
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense Totale Cumulée")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def plot_q_sensitivity(q_value_history, log_interval, state_to_track_str, action_to_track_str, title="2. Courbe de Sensibilité aux Épisodes (Convergence de la Q-Value)"):
    if not q_value_history:
        print("L'historique des Q-values est vide. Impossible de tracer la sensibilité.")
        return
        
    episodes = np.arange(log_interval, len(q_value_history) * log_interval + log_interval, log_interval)
    
    if len(episodes) > len(q_value_history):
        episodes = episodes[:len(q_value_history)]
    elif len(episodes) < len(q_value_history):
        episodes = np.append(episodes, episodes[-1] + log_interval) 
        
    if len(q_value_history) > 0:
        episodes[-1] = (len(q_value_history) - 1) * log_interval + 1

    plt.figure(figsize=(10, 5))
    
    plt.plot(episodes, q_value_history, color='purple', linewidth=2)
    
    plt.title(title)
    plt.suptitle(f"Q-Value tracée pour: $Q(S_0, A)$ avec $S_0$=Agent en {state_to_track_str} et A={action_to_track_str}", fontsize=10)
    plt.xlabel(f"Épisode (Log: {log_interval} ép.)")
    plt.ylabel("Valeur $Q$ (pour l'état-action tracé)")
    plt.grid(True)
    plt.show(block=False)

# --- Sensibilité aux grilles ---
def analyze_grid_sensitivity(grid_sizes, num_episodes, obstacles):
    """Effectue l'entraînement pour plusieurs tailles de grille et analyse les résultats."""
    
    final_reward_means = []
    
    print("\n" + "="*50)
    print("ANALYSE DE LA SENSIBILITÉ PAR RAPPORT À LA TAILLE DE LA GRILLE")
    print("="*50)

    # Nous allons nous baser sur les 10% dernières récompenses
    window = max(1, num_episodes // 10) 
    
    for size in grid_sizes:
        print(f"\n***** Démarrage pour Grille {size}x{size} *****")
        
        # Créer un nouvel environnement et un nouvel agent pour chaque taille
        env = GymnasiumGridWorld(grid_size=size, obstacles=obstacles, render_mode=None)
        agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=1.0)
        
        # Entraînement
        reward_history, _, _ = train_agent(env, agent, num_episodes=num_episodes)
        env.close()
        
        # Calcul de la moyenne des récompenses sur la dernière fenêtre
        if len(reward_history) >= window:
            mean_reward = np.mean(reward_history[-window:])
        else:
            mean_reward = np.mean(reward_history)
            
        final_reward_means.append(mean_reward)
        print(f"Grille {size}x{size} - Récompense moyenne finale: {mean_reward:.2f}")

    return grid_sizes, final_reward_means

def plot_grid_sensitivity(grid_sizes, final_reward_means, title="3. Sensibilité à la Taille de la Grille (Récompense Moyenne Finale)"):
    """Trace la sensibilité des performances par rapport à la taille de la grille."""
    
    plt.figure(figsize=(10, 5))
    plt.plot(grid_sizes, final_reward_means, marker='o', linestyle='-', color='red')
    
    plt.title(title)
    plt.xlabel("Taille de la Grille (N)")
    plt.ylabel(f"Récompense Moyenne Finale (sur 10% derniers épisodes)")
    plt.grid(True)
    plt.xticks(grid_sizes)
    plt.show(block=False)

def run_episode(env, agent, max_steps=100, save_gif=False, gif_path=None):
    """Exécute un épisode en mode pure exploitation, avec rendu SEULEMENT."""
    
    observation, info = env.reset() 
    env.total_reward = 0 
    
    done = False
    truncated = False
    step_count = 0
    
    if env.render_mode == "human":
        env.render()
        
    while not done and not truncated:
        step_count += 1
        
        if step_count >= max_steps:
             truncated = True
        
        # Exploitation pure (epsilon=0.0)
        # Nécessite de calculer le max Q à la volée, car nous utilisons un dict pour Q
        q_values = [agent._get_q(observation, a) for a in range(env.action_space.n)]
        action = np.argmax(q_values)
        
        observation, reward, done, truncated_step, info = env.step(action) 
        truncated = truncated or truncated_step
        
        if env.render_mode == "human":
            env.render()
        
    print("\n================================")
    print(f"Épisode Terminé en {step_count} pas. Récompense Totale: {env.total_reward:.1f}")
    print("================================\n")
    
    env.close()

# ====================================================================
# IV. Exécution Principale (MODIFIÉE)
# ====================================================================

if __name__ == "__main__":
    
    plt.ion()
    
    # --- Configuration Commune ---
    FIXED_OBSTACLES = [(2, 2), (4, 4), (5, 1)]
    NUM_TRAIN_EPISODES = 3000 # Réduit pour la démo multi-grilles
    
    # --- Sensibilité aux Grilles ---
    # Définir les tailles de grille à tester
    GRID_SIZES_TO_TEST = [4, 6, 8, 10]
    
    # Exécuter l'analyse de sensibilité
    grid_sizes_results, final_reward_means_results = analyze_grid_sensitivity(
        GRID_SIZES_TO_TEST, 
        NUM_TRAIN_EPISODES, 
        FIXED_OBSTACLES
    )

    # Afficher la courbe de sensibilité aux grilles
    plot_grid_sensitivity(
        grid_sizes_results, 
        final_reward_means_results,
        title="3. Sensibilité à la Taille de la Grille (Récompense Moyenne Finale)"
    )
    
    # --- Convergence et Sensibilité aux Épisodes (pour la dernière grille testée) ---
    # Ré-entraîner et tester le plus grand modèle pour la visualisation détaillée
    
    GRID_SIZE_FINAL = GRID_SIZES_TO_TEST[-1]
    
    print("\n" + "="*50)
    print(f"ENTRAÎNEMENT FINAL (Grille {GRID_SIZE_FINAL}x{GRID_SIZE_FINAL}) pour l'affichage des courbes")
    print("="*50)

    env_final_train = GymnasiumGridWorld(grid_size=GRID_SIZE_FINAL, obstacles=FIXED_OBSTACLES, render_mode=None)
    agent_final = QLearningAgent(env_final_train, alpha=0.1, gamma=0.9, epsilon=1.0)
    
    reward_history, q_value_history, log_interval = train_agent(env_final_train, agent_final, num_episodes=NUM_TRAIN_EPISODES)
    env_final_train.close()
    
    # Courbe 1: Convergence (Récompenses)
    plot_convergence(
        reward_history, 
        title=f"1. Courbe de Convergence (Récompense Totale, Grille {GRID_SIZE_FINAL}x{GRID_SIZE_FINAL})"
    )

    # Courbe 2: Sensibilité aux Épisodes (Q-Value)
    r_g_fixed, c_g_fixed = GRID_SIZE_FINAL - 1, GRID_SIZE_FINAL - 1
    state_str = f"(0, 0, {r_g_fixed}, {c_g_fixed})"
    action_str = "DOWN (1)"
    plot_q_sensitivity(
        q_value_history, 
        log_interval, 
        state_str, 
        action_str,
        title=f"2. Sensibilité aux Épisodes (Convergence Q-Value, Grille {GRID_SIZE_FINAL}x{GRID_SIZE_FINAL})"
    )
    
    # Test de l'Agent Entraîné (Exploitation)
    print("\n--- TEST FINAL: Agent Q-Learning (Exploitation) avec Rendu ---")
    env_run_q = GymnasiumGridWorld(grid_size=GRID_SIZE_FINAL, obstacles=FIXED_OBSTACLES, render_mode="human")
    agent_final.epsilon = 0.0 # Exploitation pure
    
    run_episode(
        env_run_q, 
        agent_final, 
        max_steps=GRID_SIZE_FINAL * GRID_SIZE_FINAL, 
        save_gif=False
    )

    print("\n--- FIN DU PROGRAMME ---")
    
    plt.ioff()
    plt.show(block=True)