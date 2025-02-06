import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Inizializza pygame
pygame.init()

# Dimensioni della finestra
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doom-like Game with AI")

# Colori
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Clock per il controllo del frame rate
clock = pygame.time.Clock()

# Classe per il giocatore
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT - 50)
        self.speed = 5

    def update(self, action):
        if action == 0 and self.rect.left > 0:  # Sinistra
            self.rect.x -= self.speed
        elif action == 1 and self.rect.right < WIDTH:  # Destra
            self.rect.x += self.speed

# Classe per i nemici
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIDTH - self.rect.width)
        self.rect.y = random.randint(-100, -40)
        self.speed = random.randint(2, 5)

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(0, WIDTH - self.rect.width)
            self.speed = random.randint(2, 5)

# Classe per i proiettili
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = 7

    def update(self):
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.kill()

# Rete neurale per l'apprendimento
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Parametri del reinforcement learning
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

# Ambiente di gioco
player = Player()
enemies = pygame.sprite.Group()
bullets = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
all_sprites.add(player)

for _ in range(8):
    enemy = Enemy()
    all_sprites.add(enemy)
    enemies.add(enemy)

# Funzione per ottenere lo stato corrente del gioco
def get_state():
    player_x = player.rect.x / WIDTH
    enemy_states = []
    for enemy in enemies:
        enemy_states.append(enemy.rect.x / WIDTH)
        enemy_states.append(enemy.rect.y / HEIGHT)
    while len(enemy_states) < 16:  # Assicurati che ci siano sempre 16 valori
        enemy_states.extend([0, 0])
    return np.array([player_x] + enemy_states[:16])

# Inizializzazione del modello e dell'ottimizzatore
input_dim = 1 + 16  # Posizione del giocatore + stati dei nemici
output_dim = 3  # Azioni: sinistra, destra, non fare nulla
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=MEMORY_SIZE)

steps_done = 0

# Funzione per selezionare un'azione
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * EPS_DECAY ** steps_done
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
    else:
        return random.randrange(output_dim)

# Funzione per ottimizzare il modello
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*transitions))
    
    # Convert the list of numpy.ndarrays to a single numpy.ndarray
    state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    
    current_q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + GAMMA * next_q_values
    
    loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Statistiche globali
total_deaths = 0
total_attempts = 0

# Esegui un singolo tentativo (episodio)
def run_episode():
    global total_deaths, total_attempts
    total_attempts += 1
    total_reward = 0
    state = get_state()

    # Timer per limitare il tasso di fuoco
    last_shot_time = pygame.time.get_ticks()
    SHOOT_INTERVAL = 1000  # 1 proiettile al secondo

    running = True
    while running:
        # Seleziona un'azione per il giocatore
        action = select_action(state)
        player.update(action)

        # Aggiorna i nemici e i proiettili
        enemies.update()
        bullets.update()

        # Sparo con limite di frequenza
        current_time = pygame.time.get_ticks()
        if current_time - last_shot_time >= SHOOT_INTERVAL:
            bullet = Bullet(player.rect.centerx, player.rect.top)
            all_sprites.add(bullet)
            bullets.add(bullet)
            last_shot_time = current_time

        # Controlla le collisioni
        hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
        for hit in hits:
            enemy = Enemy()
            all_sprites.add(enemy)
            enemies.add(enemy)
            total_reward += 10

        if pygame.sprite.spritecollideany(player, enemies):
            total_reward -= 100
            total_deaths += 1
            running = False

        # Calcola la ricompensa e aggiorna la memoria
        reward = total_reward
        next_state = get_state()
        memory.append((state, action, reward, next_state))
        state = next_state

        optimize_model()

        # Disegna
        screen.fill(BLACK)
        all_sprites.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    # Stampa statistiche
    print(f"Episode {total_attempts}, Total Reward: {total_reward}")
    if total_deaths > 0:
        score_to_death_ratio = total_reward / total_deaths
        print(f"Score-to-Death Ratio: {score_to_death_ratio:.2f}")
# Funzione per resettare il gioco
def reset_game():
    global player, enemies, bullets, all_sprites
    # Resetta il giocatore
    player.rect.center = (WIDTH // 2, HEIGHT - 50)
    # Rimuovi tutti i nemici e i proiettili
    enemies.empty()
    bullets.empty()
    all_sprites.empty()
    all_sprites.add(player)
    # Rigenera i nemici
    for _ in range(8):
        enemy = Enemy()
        all_sprites.add(enemy)
        enemies.add(enemy)

# Ciclo principale per eseguire episodi continui
running = True
while running:
    # Esegui un singolo episodio
    run_episode()
    
    # Resetta il gioco per il prossimo episodio
    reset_game()
    
    # Controlla gli eventi per uscire dal gioco
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

# Esci da pygame
pygame.quit()

# Esegui un singolo episodio
run_episode()

pygame.quit()
