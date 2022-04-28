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
        self.wind = 10
        self.isopen = True
        self.boat_width = 10
        self.boat_height = 30
        self.reward = 0
        self.screen = None
        self.step_ctr = 0
        self.trial_path = [ [] for _ in range(500) ] # Will not be reset after every trial - stores the path history of each episode
        # When to fail the episode
        self.x_lower_bound = 0
        self.x_upper_bound = 600
        self.y_lower_bound = 0
        self.y_upper_bound = 600
        # When to terminate the episode successfully
        self.x_target = 300
        self.y_target = 30
        # TODO - check if this works correctly
        self.action_space = spaces.Discrete(2)
        low = np.array([0, 0, 0, 0,], dtype=np.float32,)
        high = np.array([self.x_upper_bound, self.y_upper_bound, 360, 30,],dtype=np.float32,) #TODO - check theta and velocity upper bounds
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

    def step(self, action, trial_no):
        # Execute one time step within the environment
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.step_ctr += 1
        x, y, theta, velocity = self.state
        prev_x = x
        prev_y = y

        theta_change = [-0.1, 0.1][action]
        theta = theta + theta_change

        velocity = self.wind*(1 - np.exp(-(theta ** 2) / (np.pi / 2)))
        x = int(x + velocity*math.cos(theta))
        y = int(y - velocity*math.sin(theta))

        # Store the path
        self.trial_path[trial_no].append([x, y])

        self.state = (x, y, theta, velocity)
        print(self.state, ' -> ', theta_change)

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

        if done_unsuccessfully:
            self.reward -= 0.1
        elif done_successfully:
            self.reward += 10

        if not(done_unsuccessfully or done_successfully) and self.step_ctr % 20 == 0:
            # rewards for moving closer/farther from the target after every 100 steps
            if (((self.x_target - prev_x)**2 + (self.y_target - prev_y)**2) > ((self.x_target - x)**2 + (self.y_target - y)**2)): #moved closer to target
                self.reward += (1 / math.sqrt((prev_x - x)**2 + (prev_y - y)**2))
            elif (((self.x_target - prev_x)**2 + (self.y_target - prev_y)**2) < ((self.x_target - x)**2 + (self.y_target - y)**2)): #moved farther away from target
                self.reward -= (1 / math.sqrt((prev_x - x)**2 + (prev_y - y)**2))

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
        boat_img = pygame.transform.rotate(boat_img, ((-180 * self.state[2]) / np.pi))
        self.screen.blit(boat_img, (self.state[0] - (self.boat_width / 2), self.state[1] - (self.boat_height / 2)))

        # denoting the path as a trail
        pixel_array = pygame.PixelArray(self.screen)
        for trial, path in enumerate(self.trial_path):
            if trial < trial_no:
                for step in path:
                    if step[0] > 0 and step[1] > 0 and step[0] < 600 and step[1] < 600:
                        pixel_array[int(step[0]), int(step[1])] = (50, 50, 50)
                        #self.screen.set_at((path[step][0], path[step][1]), (150, 150, 150)) # very slow to render

        pixel_array.close()
        pygame.display.update()

        if mode == "human":
            pygame.display.flip()
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
