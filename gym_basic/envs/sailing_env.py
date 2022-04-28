import gym
from gym import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import math

class SailingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SailingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.theta = 0
        self.init_x = 300
        self.init_y = 500
        self.x = 300
        self.y = 500
        self.velocity = 0
        self.drag = 0.05
        self.wind = 10
        self.isopen = True
        self.boat_width = 10
        self.boat_height = 30
        self.reward = 0
        self.screen = None
        self.step_ctr = 0
        # When to fail the episode
        self.x_lower_bound = 0
        self.x_upper_bound = 600
        self.y_lower_bound = 0
        self.y_upper_bound = 600
        # When to terminate the episode successfully
        self.x_target = 50
        self.y_target = 100
        # TODO - check if this works correctly
        self.action_space = spaces.Discrete(31)
        low = np.array([0, 0, 0, 0,], dtype=np.float32,)
        high = np.array([self.x_upper_bound, self.y_upper_bound, 360, 30,],dtype=np.float32,) #TODO - check theta and velocity upper bounds
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

    def step(self, action):
        # Execute one time step within the environment
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.step_ctr += 1
        x, y, theta, velocity = self.state
        prev_x = x
        prev_y = y

        actual_theta_diff = theta + (action - 15)
        sign = np.sign(actual_theta_diff) if actual_theta_diff != 0.0 else 1
        theta = (math.fabs(actual_theta_diff) % 360) * sign
        #print('Prev_theta: ', self.state[2], ', curr_theta: ', theta, ', action: ', action)
        velocity = velocity + self.drag*self.wind*math.sin(math.radians(theta))
        x = x + velocity*math.cos(math.radians(theta))
        y = y + velocity*math.sin(math.radians(theta))

        self.state = (x, y, theta, velocity)
        print(self.state, ' -> ', (action - 5))

        done_unsuccessfully = bool(
            x < self.x_lower_bound
            or x > self.x_upper_bound
            or y < self.y_lower_bound
            or y > self.y_upper_bound
        )
        done_successfully = bool(
            math.fabs(x - self.x_target) <= 0.01
            and math.fabs(y - self.y_target) <= 0.01
        )
        self.reward -= 0.01
        if done_unsuccessfully:
            self.reward -= 0.1
        elif done_successfully:
            self.reward += 100

        if not(done_unsuccessfully or done_successfully) and self.step_ctr % 20 == 0:
            # rewards for moving closer/farther from the target after every 100 steps
            if (((self.x_target - prev_x)**2 + (self.y_target - prev_y)**2) > ((self.x_target - x)**2 + (self.y_target - y)**2)):
                self.reward += (10 / math.sqrt((prev_x - x)**2 + (prev_y - y)**2))
            else:
                self.reward -= (10 / math.sqrt((prev_x - x)**2 + (prev_y - y)**2))

        debug_msg = "Step #"+str(self.step_ctr)
        return np.array(self.state, dtype=np.float32), self.reward, bool(done_successfully or done_unsuccessfully), {}


    def reset(self):
        # Reset the state of the environment to an initial state
        # super().reset()
        self.step_ctr = 0
        self.state = (self.init_x, self.init_y, 0, 0)
        return np.array(self.state, dtype=np.float32)

    def render(self, trial_no, mode='human'):
        # Render the environment to the screen
        screen_width = 600
        screen_height = 600
        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("The incredible voyage - trial " + str(trial_no))
        self.screen.fill((56, 185, 224))
        # draw the target
        gfxdraw.filled_circle(self.screen, int(self.x_target), int(self.y_target), 10, (98, 194, 39))
        # display the boat
        boat_img = pygame.image.load(
            r'D:/MAI/Semester 2/ATCI/Projects/openai-playground/gym/envs/classic_control/assets/boat.svg')
        boat_img = pygame.transform.rotate(boat_img, self.state[2])
        self.screen.blit(boat_img, (self.state[0] - (self.boat_width / 2), self.state[1] - (self.boat_height / 2)))
        pygame.display.update()
        # if you want to denote the path as a trail - TODO

        if mode == "human":
            pygame.display.flip()
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
