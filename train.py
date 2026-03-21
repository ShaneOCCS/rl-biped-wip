from stable_baselines3 import PPO
from environment import BipedEnv

# instance of BipedEnv
def main():
    env = BipedEnv()
    # create PPO (proximal policy optimization) model with a standard neural network, trained on louis's environment
    # verbose=1 prints training progress to the terminal
    model = PPO("MlpPolicy", env, verbose=1)
