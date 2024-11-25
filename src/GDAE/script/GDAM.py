# GDAM.py

from GDAM_env import ImplementEnv
import tensorflow as tf
import tflearn
import numpy as np
from GDAM_args import d_args
import rclpy
import time


class ActorNetwork(object):

    def __init__(self, sess, logger):
        self.sess = sess
        self.logger = logger
        self.logger.info("Initializing ActorNetwork...")
        self.inputs, self.out, self.scaled_out, self.im_out = self.create_actor_network()
        self.network_params = tf.compat.v1.trainable_variables()
        self.logger.debug(f"ActorNetwork initialized with {len(self.network_params)} parameters.")

    def create_actor_network(self):
        self.logger.debug("Creating actor network architecture...")
        inputs = tflearn.input_data(shape=[None, 23])
        net = tflearn.fully_connected(inputs, 800)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 600)
        net = tflearn.activations.relu(net)
        im_out = inputs

        out = tflearn.fully_connected(net, 2, activation='tanh')

        scaled_out = tf.multiply(out, [1.0, 1.0])  # Ensure float types
        self.logger.debug("Actor network architecture created.")
        return inputs, out, scaled_out, im_out

    def predict(self, inputs):
        self.logger.debug("Predicting action from actor network...")
        actions = self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
        self.logger.debug(f"Predicted actions: {actions}")
        return actions


def test(env, actor):
    logger = env.get_logger()
    logger.info("Starting testing loop...")
    
    while rclpy.ok():
        action = [0.0, 0.0]
        logger.debug(f"Initial action: {action}")
        s2, toGoal = env.step(action)
        
        if len(s2) == 0:
            logger.warning("Received empty laser scan data. Sleeping for 0.1 seconds...")
            time.sleep(0.1)
            continue
        
        s = np.append(s2, toGoal)
        s = np.append(s, action)
        logger.debug(f"Initial state: {s}")

        while rclpy.ok():
            try:
                a = actor.predict([s])
                logger.debug(f"Action from actor network: {a}")
                aIn = a.copy()
                aIn[0, 0] = (aIn[0, 0] + 1.0) / 4.0
                logger.debug(f"Modified action: {aIn[0]}")
                
                s2, toGoal = env.step(aIn[0])
                logger.debug(f"New state: {s2}, ToGoal: {toGoal}")
                
                if len(s2) == 0:
                    logger.warning("Received empty laser scan data during loop. Breaking inner loop...")
                    break
                
                s = np.append(s2, a[0])
                s = np.append(s, toGoal)
                logger.debug(f"Updated state: {s}")
                
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
    
    # Initialize TensorFlow session
    sess = tf.compat.v1.InteractiveSession()
    logger.info("TensorFlow session started.")
    
    # Create ActorNetwork
    actor = ActorNetwork(sess, logger)
    
    # Initialize TensorFlow variables
    logger.info("Initializing TensorFlow variables...")
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    logger.debug("TensorFlow variables initialized.")
    
    # Enumerate and log network parameters
    a_net_params = []
    for variable in tf.compat.v1.trainable_variables():
        a_net_params.append(variable)
    
    logger.info(f"Total trainable variables in ActorNetwork: {len(a_net_params)}")
    for idx, v in enumerate(a_net_params):
        logger.debug(f"Variable {idx:3}: {v.get_shape()} - {v.name}")
    
    # Restore model from checkpoint
    saver = tf.compat.v1.train.Saver(a_net_params, max_to_keep=1)
    checkpoint_path = "/home/reinis/gym-gazebo/gym_gazebo/envs/mod/demo"
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            logger.info(f"Model successfully restored from: {checkpoint.model_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to restore model from checkpoint: {e}")
            # Optionally, exit if model restoration is critical
            # logger.error("Exiting due to failed model restoration.")
            # exit(1)
    else:
        logger.error("No checkpoint found at specified path.")
        # Optionally, proceed without restoring the model
        # Or exit the program if a trained model is required
        # Uncomment the following line to exit if no checkpoint is found
        # logger.error("Exiting due to missing checkpoint.")
        # exit(1)
    
    # Start testing loop
    try:
        test(env, actor)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected exception in main: {e}")
    finally:
        env.destroy_node()
        sess.close()
        logger.info("Environment node destroyed and TensorFlow session closed.")
        rclpy.shutdown()
        logger.info("ROS 2 shutdown completed.")


if __name__ == '__main__':
    main()
