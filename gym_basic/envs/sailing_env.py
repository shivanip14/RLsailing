import gym
from gym import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import math
from ..config.world_config import WIND_VELOCITY, WIND_DIRECTION, SCREEN_SIZE_X, SCREEN_SIZE_Y, INIT_X, INIT_Y, TARGET_X, TARGET_Y, TARGET_RADIUS, MAX_TEST_TRIALS

class SailingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SailingEnv, self).__init__()
        self.theta = 0
        self.x = INIT_X
        self.y = INIT_Y
        self.velocity = 0
        self.isopen = True
        self.boat_width = 10
        self.boat_height = 30

        self.screen = None
        self.step_ctr = 0
        self.trial_path = [[] for _ in range(500)] # Will not be reset after every trial - stores the path history of each episode
        # When to fail the episode

        # TODO - check if this works correctly
        self.action_space = spaces.Discrete(2)
        low = np.array([0, 0, -np.pi/2], dtype=np.float32,)
        high = np.array([SCREEN_SIZE_X, SCREEN_SIZE_Y, np.pi/2],dtype=np.float32,)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

    def step(self, action, trial_no=0):
        # Execute one time step within the environment
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.step_ctr += 1
        x, y, theta = self.state
        prev_x = x
        prev_y = y

        theta_change = [-0.1, 0.1][action]
        theta += theta_change

        velocity = WIND_VELOCITY*(1 - np.exp(-((theta - WIND_DIRECTION) ** 2) / (np.pi / 2)))
        x = int(x + velocity*math.sin(theta))
        y = int(y - velocity*math.cos(theta))

        # Store the path
        self.trial_path[trial_no].append([x, y])

        self.state = (x, y, theta)
        print(self.state, ' -> ', theta_change)

        channel_hit = bool(
            x < 0 or x > SCREEN_SIZE_X
        )
        target_escaped = bool(
            y < 0
            or y > SCREEN_SIZE_Y
        )
        boat_turned = bool(
            math.fabs(math.fabs(theta)-math.fabs(np.pi/2)) <= 0.01
        )
        done_unsuccessfully = bool(
            channel_hit or target_escaped or boat_turned
        )
        done_successfully = bool(
            math.fabs(x - TARGET_X) <= TARGET_RADIUS
            and math.fabs(y - TARGET_Y) <= TARGET_RADIUS
        )

        reward = -0.1

        if channel_hit or target_escaped:
            reward -= 10
        elif boat_turned:
            reward -= 5
        elif done_successfully:
            reward += 100

        if not(done_unsuccessfully or done_successfully) and self.step_ctr % 10 == 0:
            # rewards for moving closer/farther from the target after every 50 steps. No reward if no change in distance
            if (((TARGET_X - prev_x)**2 + (TARGET_Y - prev_y)**2) > ((TARGET_X - x)**2 + (TARGET_Y - y)**2)): # moved closer to target
                reward += 2*(math.sqrt((prev_x - x)**2 + (prev_y - y)**2))
            elif (((TARGET_X - prev_x)**2 + (TARGET_Y - prev_y)**2) < ((TARGET_X - x)**2 + (TARGET_Y - y)**2)): # moved farther away from target
                reward -= (math.sqrt((prev_x - x)**2 + (prev_y - y)**2))

        debug_msg = "Step #"+str(self.step_ctr)
        return np.array(self.state, dtype=np.float32), reward, bool(done_successfully or done_unsuccessfully), {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # super().reset()
        self.step_ctr = 0
        self.state = (INIT_X, INIT_Y, 0)
        return np.array(self.state, dtype=np.float32)

    def render(self, trial_no=0, highest_reward_trial_no=0, highest_reward=0, max_trials=MAX_TEST_TRIALS, mode='human'):
        # Render the environment to the screen
        screen_width = 600
        screen_height = 600
        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("The incredible voyage - trial " + str(trial_no))
        self.screen.fill((56, 185, 224))
        # draw the target
        gfxdraw.filled_circle(self.screen, int(TARGET_X), int(TARGET_Y), TARGET_RADIUS, (98, 194, 39))

        # display the prevalent wind conditions
        wind_img = pygame.image.load(r'D:/MAI/Semester 2/ATCI/Projects/openai-sailing/assets/wind.png')
        wind_img = pygame.transform.rotate(wind_img, ((-180 * WIND_DIRECTION) / np.pi))
        display_font = pygame.font.SysFont('calibri', 14)
        wind_indicator_text = display_font.render(str(WIND_VELOCITY)+'kts', True, (0, 0, 0))
        self.screen.blit(wind_img, (500, 50))
        self.screen.blit(wind_indicator_text, (515, 30))

        # display the boat
        boat_img = pygame.image.load(r'D:/MAI/Semester 2/ATCI/Projects/openai-sailing/assets/boat.svg')
        boat_img = pygame.transform.rotate(boat_img, ((-180 * self.state[2]) / np.pi))
        self.screen.blit(boat_img, (self.state[0] - (self.boat_width / 2), self.state[1] - (self.boat_height / 2)))

        # denoting the path as a trail
        pixel_array = pygame.PixelArray(self.screen)
        for trial, path in enumerate(self.trial_path):
            if trial < trial_no:
                for step in path:
                    if step[0] > 0 and step[1] > 0 and step[0] < 600 and step[1] < 600:
                        if trial == highest_reward_trial_no:
                            pixel_array[int(step[0]), int(step[1])] = (255, 255, 255)
                        else:
                            pixel_array[int(step[0]), int(step[1])] = (50, 50, 50)
                        #self.screen.set_at((path[step][0], path[step][1]), (150, 150, 150)) # very slow to render

        pixel_array.close()
        pygame.display.update()

        if trial_no == max_trials - 1:
            pygame.image.save(self.screen, "gym_basic/results/vw" + str(WIND_VELOCITY) + "_wd" + str(WIND_DIRECTION) + "_hr" + str(highest_reward) + ".jpg")

        if mode == "human":
            pygame.display.flip()
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
