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

#hyper paramerters
num_of_steps = 200
num_of_episodes = 1
num_of_generations = 100
is_render = False
#Cart Pole
# config_path = 'properties/CartPole-v0/config'
# env = gym.make('CartPole-v0')

#Mountain Car
config_path = 'properties/MountainCar-v0/config'
env = gym.make('MountainCar-v0')

def evaluation(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = do_rollout(net, is_render)


def do_rollout(agent, render=False):
    total_reward = 0
    ob = env.reset()
    t = 0
    for t in range(num_of_steps):
        outputs = agent.activate(ob)
        a = np.argmax(outputs)
        (ob, reward, done, _info) = env.step(a)
        total_reward += reward
        if render and t%3==0: env.render()
        if done: break
    return total_reward

def run(config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

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