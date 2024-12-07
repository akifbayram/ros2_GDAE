# actor_integration.py

from script.GDAM_env import ImplementEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from script.GDAM_args import d_args
import rclpy
import time


class ActorNetwork(nn.Module):
    def __init__(self, logger):
        super(ActorNetwork, self).__init__()
        self.logger = logger
        self.logger.info("Initializing PyTorch ActorNetwork...")

        # Define the network architecture matching the saved model's structure
        self.trunk = nn.Sequential(
            nn.Linear(25, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),  
            nn.Tanh()
        )

        # Initialize weights (optional but recommended)
        self._initialize_weights()

        self.logger.debug("PyTorch ActorNetwork architecture created.")

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        self.logger.debug("Forward pass through ActorNetwork...")
        return self.trunk(x)

    def predict(self, inputs):
        self.logger.debug("Predicting action from actor network...")
        self.eval() 
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(np.array(inputs)).float()
            actions = self.forward(inputs_tensor).numpy()
        self.logger.debug(f"Predicted actions: {actions}")
        return actions


def test(env, actor):
    logger = env.get_logger()
    logger.info("Starting testing loop...")

    while rclpy.ok():
        action = [0.0, 0.0, 0.0, 0.0] 
        logger.debug(f"Initial action: {action}")
        s2, toGoal = env.step(action)

        if len(s2) == 0:
            logger.warning("Received empty laser scan data. Sleeping for 0.1 seconds...")
            time.sleep(0.1)
            continue

        s = np.append(s2, toGoal)
        s = np.append(s, action)
        logger.debug(f"Initial state: {s}")

        if len(s) != 25:
            logger.error(f"State vector has incorrect size: {len(s)}. Expected 25.")
            time.sleep(0.1)
            continue

        while rclpy.ok():
            try:
                a = actor.predict([s])
                logger.debug(f"Action from actor network: {a}")

                aIn = a.copy()
                for i in range(aIn.shape[1]):
                    aIn[0, i] = (aIn[0, i] + 1.0) / 4.0  # Example scaling
                logger.debug(f"Modified action: {aIn[0]}")

                s2, toGoal = env.step(aIn[0].tolist())  # Ensure action is a list

                logger.debug(f"New state: {s2}, ToGoal: {toGoal}")

                if len(s2) == 0:
                    logger.warning("Received empty laser scan data during loop. Breaking inner loop...")
                    break

                s = np.append(s2, toGoal)
                s = np.append(s, a[0])
                logger.debug(f"Updated state: {s}")

                if len(s) != 25:
                    logger.error(f"Updated state vector has incorrect size: {len(s)}. Expected 25.")
                    break

                rclpy.spin_once(env, timeout_sec=0.1)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Exception in testing loop: {e}")
                break


def main():
    rclpy.init()

    # Initialize environment
    env = ImplementEnv(d_args)
    logger = env.get_logger()
    logger.info("GDAM environment initialized.")

    # Create ActorNetwork
    actor = ActorNetwork(logger)

    # Load the trained PyTorch Actor model
    actor_model_path = "/home/cs488/ros2_GDAE/src/GDAE/model/SAC_actor.pth"
    try:
        try:
            actor.load_state_dict(torch.load(actor_model_path, map_location=torch.device('cpu'), weights_only=True), strict=True)
            logger.info(f"PyTorch ActorNetwork successfully loaded from {actor_model_path} with weights_only=True")
        except TypeError:
            # weights_only not supported in this PyTorch version
            actor.load_state_dict(torch.load(actor_model_path, map_location=torch.device('cpu')), strict=True)
            logger.warning("'weights_only' parameter not supported in your PyTorch version. Proceeding without it.")
        actor.eval()  # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load PyTorch Actor model from {actor_model_path}: {e}")
        exit(1)

    # Start testing loop
    try:
        test(env, actor)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected exception in main: {e}")
    finally:
        env.destroy_node()
        logger.info("Environment node destroyed.")
        rclpy.shutdown()
        logger.info("ROS 2 shutdown completed.")


if __name__ == '__main__':
    main()
