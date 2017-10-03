#!/usr/bin/env python

"""Description:
NEAT works on GYM Cart Pole
Workable version for both  CartPole and MountainCar
"""

import os

import gym
import neat
import numpy as np
import visualize
import csv
from datetime import datetime

#hyper paramerters
num_of_steps = 200
num_of_episodes = 1
num_of_generations = 100
test_episodes = 1
time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
is_render = False
#Cart Pole
config_path = 'properties/CartPole-v0/config'
env = gym.make('CartPole-v0')

#Mountain Car
# config_path = 'properties/MountainCar-v0/config'
# env = gym.make('MountainCar-v0')

# Pong-ram-v0
# config_path = 'properties/MountainExtraLongCar-v0/config'
# env = gym.make('MountainExtraLongCar-v0')
print("action space: ", env.action_space)
print("observation space: ", env.observation_space)

generation = 1


def evaluation(genomes, config):
    nets = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = do_rollout(net, is_render)
        nets.append((genome, net))

    # sort the genomes by fitness
    nets_sorted = sorted(nets, key=lambda x: x[0].fitness, reverse=True)

    # save the best individual's genomes
    best_genome, best_net = nets_sorted[0]

    global generation
    test_best_agent(generation, best_net, is_render)
    generation = generation +  1

def do_rollout(agent, render=False):
    total_reward = 0
    ob = env.reset()
    t = 0
    for t in range(num_of_steps):
        outputs = agent.activate(ob)
        a = np.argmax(outputs)
        (ob, reward, done, _info) = env.step(a)
        total_reward += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_reward


def test_best_agent(generation_count, net, render=False):
    total_steps = []
    total_rewards = []

    for i in range(test_episodes):
        ob = env.reset()
        steps = 0
        rewards = 0
        while True:
            output = net.activate(ob)
            action = np.argmax(output)
            ob, reward, done, info = env.step(action)

            rewards += reward
            if render and steps % 3 == 0:
                env.render()

            steps += 1
            if done:
                break

        total_steps.append(steps)
        total_rewards.append(rewards)

    average_steps_per_episode = np.mean(np.array(total_steps))
    average_rewards_per_episode = np.mean(np.array(total_rewards))

    # save this to file along with the generation number
    entry = [generation_count, average_steps_per_episode, average_rewards_per_episode]
    with open(r'results/agent_evaluation-{0}.csv'.format(time), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(entry)


def run(config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    # p.add_reporter(neat.Checkpointer(10, 900))

    # Add a stdout reporter to show progress in the terminal.
    winner = p.run(evaluation, num_of_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    best_fitness = do_rollout(winner_net, is_render)
    print("Test fitness of the best genome: ", best_fitness)


if __name__ == '__main__':
    run(config=config_path)
