import pygame
import random
import time

# 初始化 Pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 游戏窗口尺寸
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 蛇身块大小
BLOCK_SIZE = 20

# 初始化游戏窗口
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("贪吃蛇游戏")

clock = pygame.time.Clock()

# 方向常量
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Snake:
  def __init__(self):
    self.length = 1
    self.positions = [(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)]
    self.direction = RIGHT
    self.color = GREEN

  def get_head_position(self):
    return self.positions[0]

  def move(self):
    cur_head = self.get_head_position()
    x, y = cur_head

    if self.direction == UP:
      y -= BLOCK_SIZE
    elif self.direction == DOWN:
      y += BLOCK_SIZE
    elif self.direction == LEFT:
      x -= BLOCK_SIZE
    elif self.direction == RIGHT:
      x += BLOCK_SIZE

    new_head = (x, y)

    # 插入新头部，移除尾部
    self.positions.insert(0, new_head)
    if len(self.positions) > self.length:
      self.positions.pop()

  def reset(self):
    self.length = 1
    self.positions = [(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)]
    self.direction = RIGHT

  def draw(self, surface):
    for p in self.positions:
      pygame.draw.rect(surface, self.color, (p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))


class Food:
  def __init__(self):
    self.position = (0, 0)
    self.color = RED
    self.randomize_position()

  def randomize_position(self):
    while True:
      x = random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
      y = random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
      self.position = (x, y)
      # 确保食物不在蛇身上
      if self.position not in snake.positions:
        break

  def draw(self, surface):
    pygame.draw.rect(surface, self.color, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))


def show_game_over():
  font = pygame.font.SysFont("PingFang", 72)
  text = font.render("游戏结束!", True, WHITE)
  screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))

  font = pygame.font.SysFont("PingFang", 36)
  text = font.render("按空格键重新开始", True, WHITE)
  screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 + 50))

  pygame.display.update()


def draw_score(score):
  font = pygame.font.SysFont("PingFang", 24)
  text = font.render(f"得分: {score}", True, WHITE)
  screen.blit(text, (10, 10))


snake = Snake()
food = Food()
game_over = False
score = 0

# 游戏主循环
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      if game_over:
        if event.key == pygame.K_SPACE:
          snake.reset()
          food.randomize_position()
          score = 0
          game_over = False
      else:
        if event.key == pygame.K_UP and snake.direction != DOWN:
          snake.direction = UP
        elif event.key == pygame.K_DOWN and snake.direction != UP:
          snake.direction = DOWN
        elif event.key == pygame.K_LEFT and snake.direction != RIGHT:
          snake.direction = LEFT
        elif event.key == pygame.K_RIGHT and snake.direction != LEFT:
          snake.direction = RIGHT

  if not game_over:
    snake.move()

    # 检测吃到食物
    if snake.get_head_position() == food.position:
      snake.length += 1
      score += 1
      food.randomize_position()

    # 碰撞检测
    head = snake.get_head_position()
    # 边界检测
    if head[0] < 0 or head[0] >= WINDOW_WIDTH or head[1] < 0 or head[1] >= WINDOW_HEIGHT:
      game_over = True
    # 自碰检测
    for body in snake.positions[1:]:
      if head == body:
        game_over = True

    screen.fill(BLACK)
    snake.draw(screen)
    food.draw(screen)
    draw_score(score)
  else:
    show_game_over()

  pygame.display.update()
  clock.tick(6.5)  # 控制游戏速度

pygame.quit()
