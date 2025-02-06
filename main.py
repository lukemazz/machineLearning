import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Initialize pygame
pygame.init()

# Window dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game with AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Clock for frame rate control
clock = pygame.time.Clock()

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))  # Create a green square for the player
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT - 50)  # Start at the bottom center
        self.speed = 5  # Movement speed

    def update(self, action):
        """Update the player's position based on the selected action."""
        if action == 0 and self.rect.left > 0:  # Move left
            self.rect.x -= self.speed
        elif action == 1 and self.rect.right < WIDTH:  # Move right
            self.rect.x += self.speed

# Enemy class
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))  # Create a red square for the enemy
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIDTH - self.rect.width)  # Random x position
        self.rect.y = random.randint(-100, -40)  # Start above the screen
        self.speed = random.randint(2, 5)  # Random speed

    def update(self):
        """Move the enemy downward. Respawn at the top if it goes off-screen."""
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(0, WIDTH - self.rect.width)
            self.speed = random.randint(2, 5)

# Bullet class
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))  # Create a white rectangle for the bullet
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)  # Spawn at the player's position
        self.speed = 7  # Bullet speed

    def update(self):
        """Move the bullet upward. Remove it if it goes off-screen."""
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.kill()

# Neural network for reinforcement learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, output_dim)  # Output layer

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Reinforcement learning parameters
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
EPS_START = 1.0  # Initial exploration rate
EPS_END = 0.01  # Final exploration rate
EPS_DECAY = 0.995  # Exploration decay rate
TARGET_UPDATE = 10  # Frequency of updating the target network
MEMORY_SIZE = 10000  # Size of the replay memory

# Game environment setup
player = Player()
enemies = pygame.sprite.Group()
bullets = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
all_sprites.add(player)

for _ in range(8):  # Add 8 enemies to the game
    enemy = Enemy()
    all_sprites.add(enemy)
    enemies.add(enemy)

# Function to get the current state of the game
def get_state():
    """Generate the current state representation for the AI."""
    player_x = player.rect.x / WIDTH  # Normalize player's x position
    enemy_states = []
    for enemy in enemies:
        enemy_states.append(enemy.rect.x / WIDTH)  # Normalize enemy x positions
        enemy_states.append(enemy.rect.y / HEIGHT)  # Normalize enemy y positions
    while len(enemy_states) < 16:  # Pad with zeros if there are fewer than 8 enemies
        enemy_states.extend([0, 0])
    return np.array([player_x] + enemy_states[:16])

# Initialize the model and optimizer
input_dim = 1 + 16  # Player position + enemy states
output_dim = 3  # Actions: move left, move right, do nothing
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=MEMORY_SIZE)

steps_done = 0

# Function to select an action using epsilon-greedy policy
def select_action(state):
    """Select an action based on the current state."""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * EPS_DECAY ** steps_done
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
    else:
        return random.randrange(output_dim)

# Function to optimize the model
def optimize_model():
    """Train the neural network using experience replay."""
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*transitions))

    # Convert lists of numpy arrays to tensors
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

# Global statistics
total_deaths = 0
total_attempts = 0

# Function to run a single episode
def run_episode():
    """Run one episode of the game."""
    global total_deaths, total_attempts
    total_attempts += 1
    total_reward = 0
    state = get_state()

    # Fire rate limiter
    last_shot_time = pygame.time.get_ticks()
    SHOOT_INTERVAL = 1000  # 1 bullet per second

    running = True
    while running:
        # Select and perform an action
        action = select_action(state)
        player.update(action)

        # Update enemies and bullets
        enemies.update()
        bullets.update()

        # Fire a bullet if enough time has passed
        current_time = pygame.time.get_ticks()
        if current_time - last_shot_time >= SHOOT_INTERVAL:
            bullet = Bullet(player.rect.centerx, player.rect.top)
            all_sprites.add(bullet)
            bullets.add(bullet)
            last_shot_time = current_time

        # Check for collisions
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

        # Calculate reward and store experience
        reward = total_reward
        next_state = get_state()
        memory.append((state, action, reward, next_state))
        state = next_state

        optimize_model()

        # Draw everything
        screen.fill(BLACK)
        all_sprites.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    # Print statistics
    print(f"Episode {total_attempts}, Total Reward: {total_reward}")
    if total_deaths > 0:
        score_to_death_ratio = total_reward / total_deaths
        print(f"Score-to-Death Ratio: {score_to_death_ratio:.2f}")

# Function to reset the game
def reset_game():
    """Reset the game for a new episode."""
    global player, enemies, bullets, all_sprites
    player.rect.center = (WIDTH // 2, HEIGHT - 50)
    enemies.empty()
    bullets.empty()
    all_sprites.empty()
    all_sprites.add(player)
    for _ in range(8):
        enemy = Enemy()
        all_sprites.add(enemy)
        enemies.add(enemy)

# Main loop to run continuous episodes
running = True
while running:
    run_episode()
    reset_game()

    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

pygame.quit()
