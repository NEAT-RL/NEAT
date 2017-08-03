from __future__ import print_function

import multiprocessing
import os
import pickle
import argparse
import logging
import sys
import gym.wrappers as wrappers
import matplotlib.pyplot as plt
import neat
import numpy as np
import gym
from datetime import datetime

import visualize


class Neat(object):
    def __init__(self, config):
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 10 generations or 900 seconds.
        pop.add_reporter(neat.Checkpointer(10, 900))
        self.config = config
        self.population = pop
        self.pool = multiprocessing.Pool()

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
            g.fitness = []

        for i, (genome, net) in enumerate(nets):
            # run episodes
            episode_count = 0
            MAX_EPISODES = 1
            total_score = 0
            while episode_count < MAX_EPISODES:
                state = env.reset()
                terminal_reached = False
                max_steps = 200
                step = 0
                while not terminal_reached and step < max_steps:
                    env.render()
                    # take action based on observation
                    nn_output = net.activate(state)
                    action = np.argmax(nn_output)

                    # perform next step
                    observation, reward, done, info = env.step(action)
                    total_score += reward
                    observation, reward, done, info = env.step(action)
                    total_score += reward
                    step += 1
                    if done:
                        terminal_reached = True

                episode_count += 1

            # assign fitness to be total rewards / number of episodes == steps per episodes
            genome.fitness = total_score/episode_count

        # save the best individual's genome
        genome, net = max(nets, key=lambda x: x[0].fitness)
        logger.debug("Best genome: %s", genome)
        logger.debug("Best genome fitness: %f", genome.fitness)
        # save genome
        # with open('best_genomes/gen-{0}-genome'.format(self.generation_count), 'wb') as f:
        #     pickle.dump(genome, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()

    logging.basicConfig(filename='log/log.debug-{0}.log'.format(datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")),
                        level=logging.DEBUG)
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.ERROR)

    env = gym.make(args.env_id)
    env = env.env
    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # Limit episode time steps to cut down on training time.
    # 400 steps is more than enough time to land with a winning score.
    # print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    # env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 400
    # print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '/tmp/neat-' + datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")
    # env = wrappers.Monitor(env, directory=outdir, force=True)

    # run the algorithm

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    agent = Neat(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            agent.execute_algorithm(100)

            visualize.plot_stats(agent.stats, ylog=False, view=False, filename="fitness.svg")

            mfs = sum(agent.stats.get_fitness_mean()[-20:]) / 20.0
            logger.debug("Average mean fitness over last 20 generations: %f", mfs)

            mfs = sum(agent.stats.get_fitness_stat(min)[-20:]) / 20.0
            logger.debug("Average min fitness over last 20 generations: %f", mfs)

            # Use the ten best genomes seen so far as an ensemble-ish control system.
            best_genomes = agent.stats.best_unique_genomes(10)

            # Save the winners
            for n, g in enumerate(best_genomes):
                name = 'winners\winner-{0}'.format(n)
                with open(name + '.pickle', 'wb') as f:
                    pickle.dump(g, f)

                visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                visualize.draw_net(config, g, view=False, filename=name + "-net-enabled.gv",
                                   show_disabled=False)
                visualize.draw_net(config, g, view=False, filename=name + "-net-enabled-pruned.gv",
                                   show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            logger.debug("User break.")
            break

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
