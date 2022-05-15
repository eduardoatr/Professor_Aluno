import argparse

import gym
import numpy as np

from models.Agent import Agent
from models.TSFramework import TSFramework


def cartpole(args):

    # Get the environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create the student agent
    agent = Agent(state_size, action_size, 0.1, 0.999, 0.1)

    # Create the teacher agent
    teacher = Agent(state_size, action_size, 0.0, 1, 0)
    teacher.load_model("./saved/teacher.h5")
    framework = TSFramework(teacher, args.budget)

    # Tracking variables
    scores, episodes = [], []
    score_total = 0

    # Training
    print(">>  [Start]")
    for epoch in range(args.epochs):

        # Initialize state
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:

            # Render training
            if args.show:
                env.render()

            # Get the actions for the current state
            action = agent.get_action(state)

            # Get teachers advice
            if args.advice == "early":
                action = framework.early_advising(action, state)

            elif args.advice == "importance":
                action = framework.importance_advising(action, state, args.threshold)

            elif args.advice == "mistake":
                action = framework.mistake_correcting(action, state, args.threshold)

            # Run the environment
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Generate the reward
            reward = reward if not done or score == 499 else -100

            # Save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            # Train the model
            agent.train_model()

            # Update variables
            score += reward
            state = next_state

            if done:

                # Updates the target model
                agent.update_target_model()

                # Shows the play time and total score
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(epoch)
                score_total += score
                print(f">>   Epoch {epoch + 1}: {(score_total / (epoch + 1))}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Teacher-Student Framework")
    parser.add_argument(
        "advice",
        type=str,
        help="advice approach for the teacher agent",
        choices=["early", "importance", "mistake", "none"],
    )
    parser.add_argument(
        "--show", action="store_true", help="show the environment's animation"
    )
    parser.add_argument(
        "-epochs", type=int, default=300, help="number of training epochs"
    )
    parser.add_argument("-budget", type=int, default=2000, help="teacher's budget")
    parser.add_argument("-threshold", type=int, default=10, help="importance threshold")

    arguments = parser.parse_args()
    cartpole(arguments)
