import turtle
import random
import time
import math
import gym
from gym import spaces
from gym.utils import seeding

HEAD_SIZE = 20
HEIGHT = 20      # number of steps vertically from wall to wall of screen
WIDTH = 20       # number of steps horizontally from wall to wall of screen
PIXEL_H = 20*HEIGHT  # pixel height + border on both sides
PIXEL_W = 20*WIDTH   # pixel width + border on both sides

SLEEP = 0.5     # time to wait between steps

GAME_TITLE = 'Snake'
# BG_COLOR = 'white'
BG_COLOR = tuple(i/255.0 for i in (242,225,242)) # lavender is gentler than bright white

SNAKE_SHAPE = 'square'
SNAKE_COLOR = 'black'
SNAKE_START_LOC_H = 0
SNAKE_START_LOC_V = 0

APPLE_SHAPE = 'circle'
APPLE_COLOR = 'green'

class Snake(gym.Env):

    def __init__(self, human=False, env_info={'state_space':None}):
        super(Snake, self).__init__()

        self.done = False
        self.seed()
        self.reward = 0
        self.action_space = 4 # dimension of the action space is 4 (up, down, right, left)
        self.state_space = 12
        self.total, self.maximum = 0, 0
        self.human = human
        self.env_info = env_info

        # Create the background in which the snake hunts for the apple.
        self.win = turtle.Screen()
        self.win.title(GAME_TITLE)
        self.win.bgcolor(*BG_COLOR)
        self.win.tracer(0)
        self.win.setup(width=PIXEL_W+32, height=PIXEL_H+32)

        # snake
        self.snake = turtle.Turtle()
        self.snake.shape(SNAKE_SHAPE)
        self.snake.speed(0)
        self.snake.penup()
        self.snake.color(SNAKE_COLOR)
        self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        self.snake.direction = 'stop'
        # snake body, add first element (for location of snake's head)
        self.snake_body = []
        self.add_to_body()

        # apple
        self.apple = turtle.Turtle()
        self.apple.speed(0)
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.penup()
        self.move_apple(first=True)

        # distance between apple and snake
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)

        # score
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 100)
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", align='center', font=('Courier', 18, 'normal'))

        # control
        self.win.listen()
        self.win.onkey(self.go_up, 'Up')
        self.win.onkey(self.go_right, 'Right')
        self.win.onkey(self.go_down, 'Down')
        self.win.onkey(self.go_left, 'Left')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_coordinates(self):
        apple_x = random.randint(-WIDTH/2, WIDTH/2)
        apple_y = random.randint(-HEIGHT/2, HEIGHT/2)
        return apple_x, apple_y

    def move_snake(self):
        if self.snake.direction == 'stop':
            self.reward = 0
        if self.snake.direction == 'up':
            y = self.snake.ycor()
            self.snake.sety(y + 20)
        if self.snake.direction == 'right':
            x = self.snake.xcor()
            self.snake.setx(x + 20)
        if self.snake.direction == 'down':
            y = self.snake.ycor()
            self.snake.sety(y - 20)
        if self.snake.direction == 'left':
            x = self.snake.xcor()
            self.snake.setx(x - 20)

    def go_up(self):
        if self.snake.direction != "down":
            self.snake.direction = "up"

    def go_down(self):
        if self.snake.direction != "up":
            self.snake.direction = "down"

    def go_right(self):
        if self.snake.direction != "left":
            self.snake.direction = "right"

    def go_left(self):
        if self.snake.direction != "right":
            self.snake.direction = "left"

    def move_apple(self, first=False):
        if first or self.snake.distance(self.apple) < 20:
            while True:
                self.apple.x, self.apple.y = self.random_coordinates()
                self.apple.goto(round(self.apple.x*20), round(self.apple.y*20))
                if not self.body_check_apple():
                    break
            if not first:
                self.update_score()
                self.add_to_body()
            first = False
            return True


    def update_score(self):
        self.total += 1
        if self.total >= self.maximum:
            self.maximum = self.total
        self.score.clear()
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", align='center', font=('Courier', 18, 'normal'))


    def reset_score(self):
        self.score.clear()
        self.total = 0
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", align='center', font=('Courier', 18, 'normal'))


    def add_to_body(self):
        body = turtle.Turtle()
        body.speed(0)
        body.shape('square')
        body.color('black')
        body.penup()
        self.snake_body.append(body)


    def move_snakebody(self):
        if len(self.snake_body) > 0:
            for index in range(len(self.snake_body)-1, 0, -1):
                x = self.snake_body[index-1].xcor()
                y = self.snake_body[index-1].ycor()
                self.snake_body[index].goto(x, y)

            self.snake_body[0].goto(self.snake.xcor(), self.snake.ycor())


    def measure_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)


    def body_check_snake(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body[1:]:
                if body.distance(self.snake) < 20:
                    self.reset_score()
                    return True

    def body_check_apple(self):
        if len(self.snake_body) > 0:
            for body in self.snake_body[:]:
                if body.distance(self.apple) < 20:
                    return True

    def wall_check(self):
        if self.snake.xcor() > 200 or self.snake.xcor() < -200 or self.snake.ycor() > 200 or self.snake.ycor() < -200:
            self.reset_score()
            return True

    def reset(self):
        if self.human:
            time.sleep(1)
        for body in self.snake_body:
            body.goto(1000, 1000)

        self.snake_body = []
        self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        self.snake.direction = 'stop'
        self.reward = 0
        self.total = 0
        self.done = False

        state = self.get_state()

        return state


    def run_game(self):
        reward_given = False
        self.win.update()
        self.move_snake()
        if self.move_apple():
            self.reward = 10
            reward_given = True
        self.move_snakebody()
        self.measure_distance()
        if self.body_check_snake():
            self.reward = -100
            reward_given = True
            self.done = True
            if self.human:
                self.reset()
        if self.wall_check():
            self.reward = -100
            reward_given = True
            self.done = True
            if self.human:
                self.reset()
        if not reward_given:
            if self.dist < self.prev_dist:
                self.reward = 1
            else:
                self.reward = -1
        if self.human:
            time.sleep(SLEEP)
            state = self.get_state()


    # # AI agent
    # def step(self, action):
    #     if action == 0:
    #         self.go_up()
    #     if action == 1:
    #         self.go_right()
    #     if action == 2:
    #         self.go_down()
    #     if action == 3:
    #         self.go_left()
    #     self.run_game()
    #     state = self.get_state()
    #     return state, self.reward, self.done, {}

    def step(self, action):
        if action == 0: self.go_up()
        if action == 1: self.go_right()
        if action == 2: self.go_down()
        if action == 3: self.go_left()
        self.run_game()
        return self.get_state(), self.reward, self.done


    def get_state(self):
        # snake coordinates abs
        self.snake.x, self.snake.y = self.snake.xcor()/WIDTH, self.snake.ycor()/HEIGHT
        # snake coordinates scaled 0-1
        self.snake.xsc, self.snake.ysc = self.snake.x/WIDTH+0.5, self.snake.y/HEIGHT+0.5
        # apple coordinates scaled 0-1
        self.apple.xsc, self.apple.ysc = self.apple.x/WIDTH+0.5, self.apple.y/HEIGHT+0.5

        # Check to see if a wall is next to the head and in what cardinal direction.
        wall_up=1 if self.snake.y >= HEIGHT/2 else 0
        wall_down=1 if self.snake.y <= -HEIGHT/2 else 0
        wall_right=1 if self.snake.x >= WIDTH/2 else 0
        wall_left=1 if self.snake.y <= -WIDTH/2 else 0

        # Check to see if the snake's body is in the head's immediate vicinity.
        # Here are some example states where ^ is the head and . is the tail:
        #
        # [o][1][o]     [.][1][o]       [^]
        # [1][^][1]        [^][1]       [o]
        # [.][1][o]        [1][o]       [.]
        body_up=body_down=body_right=body_left=False
        if len(self.snake_body) > 3: # [o][o]
                                     # [o][<] You need a length of 4 to eat yourself.
            for body in self.snake_body[3:]: # Only loop through the 3rd body link onward.
                if body.distance(self.snake) == HEAD_SIZE: # If the body is exactly one HEAD_SIZE unit away from the head
                    if body.ycor() < self.snake.ycor(): # if the body link is underneath the head
                        body_down = True # then confirm the link is below the head.
                    elif body.ycor() > self.snake.ycor(): # otherwise if the body link is above the head,
                        body_up = True # confirm the link is above the head.
                    if body.xcor() < self.snake.xcor():
                        body_left = True
                    elif body.xcor() > self.snake.xcor():
                        body_right = True

        # state:     apple_up,     apple_right,     apple_down,     apple_left,
        #         obstacle_up,  obstacle_right,  obstacle_down,  obstacle_left,
        #        direction_up, direction_right, direction_down, direction_left
        if self.env_info['state_space'] == 'coordinates': # Let the agent receive direct knowledge of where the apple is.
            state = [self.apple.xsc, self.apple.ysc, self.snake.xsc, self.snake.ysc, \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
        elif self.env_info['state_space'] == 'no direction': # Don't let the agent know which direction the snake is moving in.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    0, 0, 0, 0]
        elif self.env_info['state_space'] == 'no body knowledge': # The agent no longer knows if the head is immediately near a body part, only walls.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    wall_up, wall_right, wall_down, wall_left, \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
        else: # Allow the agent to know which cardinal directions the apple is in, whether the head is immediately next to walls or body links, and what direction the head is moving in.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]

        return state

    def bye(self):
        self.win.bye()



if __name__ == '__main__':
    human = True
    env = Snake(human=human)

    if human:
        while True:
            env.run_game()
