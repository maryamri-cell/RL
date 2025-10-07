# Projet: Apprentissage par Renforcement - GridWorld

## Vue d'ensemble

Ce projet implémente et compare trois algorithmes d'apprentissage par renforcement (Q-Learning et SARSA) appliqués à des environnements de navigation GridWorld. L'objectif est d'analyser la convergence, la sensibilité aux hyperparamètres et la performance selon différentes configurations.

---

## 📊 Structure du Projet

Le projet contient trois implémentations distinctes:

1. **Q-Learning_Aléatoire.py** - Q-Learning avec Goal Aléatoire (État 4D)
2. **Q-Learning_Goal_fixe.py** - Q-Learning Classique (Goal Fixe, État 2D)
3. **TD(0).py** - SARSA Linéaire (Environnement Stochastique)

---

## 🎯 Implémentation 1: Q-Learning avec Goal Aléatoire (Q-Learning_Aléatoire.py)

### Description

Cette implémentation utilise un **état à 4 dimensions** qui encode à la fois la position de l'agent ET la position du goal. Cela permet à l'agent d'apprendre une politique générale qui fonctionne peu importe où se trouve le goal.

### Caractéristiques Clés

- **Espace d'état**: (position_agent_row, position_agent_col, position_goal_row, position_goal_col)
- **Goal**: Change aléatoirement à chaque épisode
- **Stockage**: Dictionnaire Python (supporte différentes tailles de grille)
- **Algorithme**: Q-Learning (off-policy)

### Avantages et Inconvénients

**✅ Avantages:**
- Généralisation: l'agent apprend à naviguer vers n'importe quel goal
- Flexibilité: supporte plusieurs tailles de grille

**❌ Inconvénients:**
- Espace d'état très grand (croissance quadratique)
- Convergence plus lente
- Consommation mémoire importante

### Résultats Expérimentaux

#### 1. Convergence sur Grande Grille (10×10)

![Convergence Q-Learning Goal Aléatoire](chemin/vers/image3.png)

**Observations:**
- La convergence est progressive mais lente
- Variance élevée due au changement de goal à chaque épisode
- La récompense moyenne converge vers environ -20 à -50

---

#### 2. Sensibilité au Nombre d'Épisodes

![Sensibilité aux Épisodes](chemin/vers/image1.png)

**Observations:**
- La Q-value pour un état-action spécifique décroît progressivement
- Convergence observée après environ 2000-2500 épisodes
- La valeur finale stabilise autour de -3.0
- Tracé pour: Q(S₀=(0,0,9,9), A=DOWN)

---

#### 3. Sensibilité à la Taille de Grille

![Sensibilité Taille de Grille](chemin/vers/image2.png)

**Observations:**
- Performance se dégrade linéairement avec la taille de grille
- Grille 4×4: récompense moyenne ≈ -15
- Grille 10×10: récompense moyenne ≈ -130
- La complexité augmente exponentiellement avec la dimension

---

### Paramètres de Configuration

| Paramètre | Valeur |
|-----------|--------|
| Tailles testées | 4×4, 6×6, 8×6, 10×10 |
| Nombre d'épisodes | 3000 |
| Alpha (α) | 0.1 |
| Gamma (γ) | 0.9 |
| Epsilon initial | 1.0 |
| Decay rate | 0.99995 |

### Récompenses

- Déplacement normal: **-1**
- Collision obstacle: **-5**
- Atteindre le goal: **+10**

---

## 🎯 Implémentation 2: Q-Learning Classique (Q-Learning_Goal_fixe.py)

### Description

Cette implémentation utilise un **état à 2 dimensions** (position de l'agent uniquement). Le goal est fixe et fait partie implicite de l'environnement. L'approche est plus standard et efficace pour des configurations statiques.

### Caractéristiques Clés

- **Espace d'état**: (row, col)
- **Goal**: Fixe (peut être multiple)
- **Stockage**: Array NumPy 3D (efficace)
- **Algorithme**: Q-Learning standard

### Avantages et Inconvénients

**✅ Avantages:**
- Convergence rapide
- Mémoire optimisée
- Performance élevée après entraînement
- Support de multiples goals simultanés

**❌ Inconvénients:**
- Spécifique à une configuration de goal
- Nécessite réentraînement si le goal change
- Taille de grille fixe

### Résultats Expérimentaux

#### 1. Convergence Rapide (10×10, 20000 épisodes)

![Convergence Q-Learning Classique](chemin/vers/image4.png)

**Observations:**
- Convergence très rapide (< 1000 épisodes)
- Récompense moyenne stabilise près de 0
- Variance minimale après convergence
- Agent trouve la politique optimale

---

#### 2. Analyse de Sensibilité Multi-Grilles

*[Insérer image d'analyse de sensibilité avec variance]*

**Observations:**
- Variance plus faible que l'approche goal aléatoire
- Performance stable sur multiples runs
- Convergence cohérente entre différentes tailles

---

### Paramètres de Configuration

| Paramètre | Valeur |
|-----------|--------|
| Taille de grille | 10×10 |
| Goals | [(6,6), (3,3)] |
| Nombre d'épisodes | 20000 |
| Alpha (α) | 0.1 |
| Gamma (γ) | 0.9 |
| Epsilon initial | 1.0 |
| Decay rate | 0.999 |

### Fonctionnalités Spéciales

- **Sauvegarde/Chargement**: Possibilité de sauvegarder la Q-table entraînée
- **Multi-goals**: Support de plusieurs objectifs simultanés
- **Analyse statistique**: Moyenne et variance sur plusieurs runs

---

## 🎯 Implémentation 3: SARSA Linéaire (TD(0).py)

### Description

Cette implémentation utilise **l'approximation de fonction linéaire** au lieu d'une table. L'algorithme SARSA (on-policy) est utilisé dans un **environnement stochastique** où les actions peuvent dévier.

### Caractéristiques Clés

- **Espace d'état**: (row, col)
- **Représentation**: Q(s,a) = θᵀφ(s,a)
- **Features**: 5 features de base × 4 actions = 20 paramètres
- **Environnement**: Stochastique (bruit d'action)
- **Algorithme**: SARSA (on-policy)

### Features Extraites

1. **r** - Position row de l'agent
2. **c** - Position col de l'agent
3. **dist_goal** - Distance Manhattan au goal le plus proche
4. **dist_obs** - Distance Manhattan à l'obstacle le plus proche
5. **bias** - Terme constant (=1.0)

### Avantages et Inconvénients

**✅ Avantages:**
- Mémoire constante O(features)
- Généralisation entre états similaires
- Robuste au bruit
- Adaptable à de grandes grilles

**❌ Inconvénients:**
- Approximation (pas de garantie d'optimalité)
- Sensible au choix de features
- Convergence plus variable
- Design de features manuel

### Résultats Expérimentaux

#### 1. Convergence en Environnement Stochastique

![Convergence SARSA Linéaire](chemin/vers/image6.png)

**Observations:**
- Convergence progressive malgré le bruit
- Variance élevée due à la stochasticité
- Moyenne glissante montre amélioration claire
- Récompense converge vers 0 (politique quasi-optimale)

---

#### 2. Évolution des Poids θ

![Évolution des Poids Theta](chemin/vers/image5.png)

**Observations:**
- Les 20 poids évoluent différemment
- Convergence observée après ~2000-4000 épisodes
- Certains poids (θ₆, θ₁₆) montrent forte variance
- Stabilisation générale vers la fin

---

### Paramètres de Configuration

| Paramètre | Valeur |
|-----------|--------|
| Taille de grille | 7×7 |
| Goal | [(6,6)] |
| Nombre d'épisodes | 10000 |
| Alpha (α) | 0.005 |
| Gamma (γ) | 0.99 |
| Epsilon initial | 1.0 |
| Decay rate | 0.995 |
| Bruit d'action | 0.1 (10%) |

### Récompenses

- Déplacement normal: **-0.5** (plus petit)
- Collision obstacle: **-5**
- Atteindre le goal: **+10**

### Stochasticité de l'Environnement

Avec un bruit de 10%, lorsque l'agent choisit une action:
- 90% de chance: l'action est exécutée correctement
- 10% de chance: l'agent dévie perpendiculairement

---

## 📊 Tableau Comparatif des Trois Approches

| Critère | rl42.py (Q-Learning 4D) | rl4.py (Q-Learning 2D) | TD(0).py (SARSA Linéaire) |
|---------|-------------------------|------------------------|---------------------------|
| **Type d'algorithme** | Q-Learning (off-policy) | Q-Learning (off-policy) | SARSA (on-policy) |
| **Dimension d'état** | 4D | 2D | 2D |
| **Représentation** | Table (Dict) | Table (NumPy) | Fonction linéaire |
| **Mémoire** | O(states²×actions) | O(states×actions) | O(features) |
| **Goal** | Aléatoire | Fixe | Fixe |
| **Environnement** | Déterministe | Déterministe | Stochastique |
| **Généralisation** | Entre goals | Aucune | Entre états |
| **Vitesse convergence** | Lente | Rapide | Moyenne |
| **Performance finale** | Moyenne | Excellente | Bonne |
| **Complexité mémoire** | Haute | Moyenne | Basse |

---

## 🔬 Analyses Produites

### 1. Courbes de Convergence

Toutes les implémentations produisent des courbes montrant:
- Récompense totale par épisode (en bleu)
- Moyenne glissante (en orange)
- Évolution claire de l'apprentissage

### 2. Analyse de Sensibilité

#### rl42.py - Deux types:
- **Sensibilité aux épisodes**: Évolution d'une Q-value spécifique
- **Sensibilité à la taille**: Performance vs dimension de grille

#### Q-Learning_Goal_fixe.py:
- **Sensibilité multi-grilles**: Moyenne et variance sur plusieurs runs

#### TD(0).py:
- **Sensibilité des paramètres**: Évolution des 20 poids θ

---

## 🎮 Environnement GridWorld

### Description Visuelle

L'environnement est une grille 2D avec:
- **Cases blanches**: Chemin libre
- **Cases rouges**: Goal (marqué "GOAL")
- **Cases noires**: Obstacles (marqué "OBS")
- **Cercle vert**: Agent

### Actions Disponibles

| Action | Code | Effet |
|--------|------|-------|
| UP | 0 | Déplacement vers le haut (row-1) |
| DOWN | 1 | Déplacement vers le bas (row+1) |
| LEFT | 2 | Déplacement vers la gauche (col-1) |
| RIGHT | 3 | Déplacement vers la droite (col+1) |

### Règles

1. L'agent commence toujours en position (0, 0)
2. Les obstacles bloquent le passage (agent reste sur place)
3. Sortir de la grille est impossible (agent reste sur place)
4. L'épisode se termine quand le goal est atteint

---

## 📈 Interprétation des Résultats

### Q-Learning avec Goal Aléatoire (Q-Learning_Aléatoire.py)

**Forces:**
- Apprend une politique générale
- Fonctionne avec n'importe quel goal

**Faiblesses:**
- Convergence lente (>2000 épisodes)
- Performance dégradée sur grandes grilles
- Forte consommation mémoire

**Cas d'usage idéal:**
- Environnements où le goal change fréquemment
- Petites grilles (<8×8)

---

### Q-Learning Classique (Q-Learning_Goal_fixe.py)

**Forces:**
- Convergence très rapide (<1000 épisodes)
- Performance optimale
- Efficace en mémoire

**Faiblesses:**
- Spécifique à une configuration
- Nécessite réentraînement si changement

**Cas d'usage idéal:**
- Goal fixe et connu
- Recherche de performance maximale
- Production avec configuration stable

---

### SARSA Linéaire (TD(0).py)

**Forces:**
- Mémoire constante
- Généralise bien
- Robuste au bruit

**Faiblesses:**
- Approximation (non exact)
- Dépend du choix de features
- Convergence variable

**Cas d'usage idéal:**
- Grandes grilles
- Environnements bruités/stochastiques
- Contraintes mémoire

---

## 🎯 Conclusions Principales

### Impact de la Représentation d'État

1. **État 4D (Q-Learning_Aléatoire.py)**: Généralise mais coûte cher en mémoire et temps
2. **État 2D (Q-Learning_Goal_fixe.py)**: Optimal pour configuration fixe
3. **Approximation linéaire (TD(0).py)**: Compromis entre mémoire et performance

### Impact de l'Algorithme

1. **Q-Learning (off-policy)**: Apprend politique optimale indépendamment
2. **SARSA (on-policy)**: Plus prudent, adapté aux environnements dangereux

### Impact de l'Environnement

1. **Déterministe**: Convergence rapide et stable
2. **Stochastique**: Nécessite plus d'épisodes, variance élevée

---

## 📚 Références Théoriques

- **Q-Learning**: Watkins, C.J. & Dayan, P. (1992). "Q-learning"
- **SARSA**: Rummery, G.A. & Niranjan, M. (1994). "On-line Q-learning using connectionist systems"
- **Function Approximation**: Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction" (2nd Edition)
- **Gymnasium**: Framework OpenAI/Farama pour environnements RL

---

## 🛠️ Dépendances Requises

```
gymnasium >= 0.29.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

---

## 📝 Notes Techniques

### Politique d'Exploration

- **Entraînement**: ε-greedy avec décroissance exponentielle
- **Test**: ε = 0 (exploitation pure)
- **Décroissance**: ε_new = max(ε_min, ε × decay_rate)

### Conditions de Terminaison

- **Success**: Agent atteint le goal
- **Truncation**: Limite de pas atteinte ou récompense trop négative

### Update Rules

**Q-Learning**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**SARSA**:
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

**SARSA Linéaire**:
```
θ ← θ + α[r + γ θᵀφ(s',a') - θᵀφ(s,a)]φ(s,a)
```

---

## 🎓 Applications Pédagogiques

Ce projet permet de comprendre:
1. **Différence off-policy vs on-policy**
2. **Impact de la représentation d'état**
3. **Trade-off table vs approximation**
4. **Effet du bruit sur l'apprentissage**
5. **Analyse de convergence en RL**

---

## 👨‍🎓 Auteur

Projet développé à des fins éducatives pour l'étude approfondie des algorithmes d'apprentissage par renforcement.

---

**Dernière mise à jour**: Octobre 2025