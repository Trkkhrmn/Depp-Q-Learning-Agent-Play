import pygame
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

pygame.init()

WIDTH, HEIGHT = 360, 360

GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQL Game Example")

player_width, player_height = 20, 20
player_x = WIDTH // 2 - player_width // 2
player_y = HEIGHT - player_height - 10
player_speed = 5

enemy_width, enemy_height = 20, 20
enemy_speed = 3
enemies = []
enemy_spawn_timer = 0
enemy_spawn_interval = 60

score = 0
enemy_speed_increase_counter = 0
enemy_spawn_increase_counter = 0

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

state_size = 4
action_size = 2
agent = DQLAgent(state_size, action_size)

def get_state():
    player_x_normalized = player_x / WIDTH
    player_y_normalized = player_y / HEIGHT
    if len(enemies) > 0:
        enemy_x_normalized = enemies[0][0] / WIDTH
        enemy_y_normalized = enemies[0][1] / HEIGHT
    else:
        enemy_x_normalized = 0
        enemy_y_normalized = 0
    return np.array([player_x_normalized, player_y_normalized, enemy_x_normalized, enemy_y_normalized]).reshape(1, 4)

running = True
while running:
    win.fill(GREEN)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state()
    action = agent.act(state)
    if action == 0 and player_x > 0:
        player_x -= player_speed
    elif action == 1 and player_x < WIDTH - player_width:
        player_x += player_speed

    for enemy in enemies:
        enemy[1] += enemy_speed
        pygame.draw.rect(win, RED, (enemy[0], enemy[1], enemy_width, enemy_height))
        if enemy[1] + enemy_height >= player_y and enemy[1] <= player_y + player_height:
            if enemy[0] + enemy_width >= player_x and enemy[0] <= player_x + player_width:
                running = False
                print("Game Over! Score:", score)
        if enemy[1] > HEIGHT:
            enemies.remove(enemy)
            score += 1
            enemy_speed_increase_counter += 1
            if enemy_speed_increase_counter >= 5:
                enemy_speed += 1
                enemy_speed_increase_counter = 0
            enemy_spawn_increase_counter += 1
            if enemy_spawn_increase_counter >= 15:
                enemy_spawn_interval -= 5
                enemy_spawn_increase_counter = 0

    enemy_spawn_timer += 1
    if enemy_spawn_timer >= enemy_spawn_interval:
        enemy_x = random.randint(0, WIDTH - enemy_width)
        enemies.append([enemy_x, -enemy_height])
        enemy_spawn_timer = 0

    pygame.draw.rect(win, WHITE, (player_x, player_y, player_width, player_height))

    pygame.display.update()

pygame.quit()