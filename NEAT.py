from __future__ import print_function

import argparse
import configparser
import csv
import logging
import multiprocessing
import os
import pickle
import sys
from datetime import datetime

import gym
import gym.wrappers as wrappers
import neat
import numpy as np
import visualize


class Neat(object):
    def __init__(self, config):
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 10 generations or 900 seconds.
        # pop.add_reporter(neat.Checkpointer(10, 900))
        self.config = config
        self.population = pop
        self.pool = multiprocessing.Pool()
        self.generation_count = 0
        self.best_agents = []

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        t_start = datetime.now()

        nets = []
        for gid, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            g.fitness = self.perform_rollout(net)
            nets.append((g, net))

        # sort the genomes by fitness
        nets_sorted = sorted(nets, key=lambda x: x[0].fitness, reverse=True)

        # save the best individual's genomes
        best_genome, best_net = nets_sorted[0]
        self.best_agents.append((self.generation_count, best_genome, best_net))
        logger.debug("Best genome fitness: %f", best_genome.fitness)

        worst_genome, worst_net = nets_sorted[len(nets_sorted) - 1]
        logger.debug("Worst genome fitness: %f", worst_genome.fitness)

        # save genome to file
        with open('best_genomes/gen-{0}-genome'.format(self.generation_count), 'wb') as f:
            pickle.dump(best_genome, f)

        logger.debug("Completed generation: %d. Time taken: %f", self.generation_count,
                     (datetime.now() - t_start).total_seconds())
        self.generation_count += 1

    @staticmethod
    def perform_rollout(net):
        # run episodes
        state = env.reset()
        terminal_reached = False
        step_size = props.getint('initialisation', 'step_size')
        max_steps = props.getint('initialisation', 'max_steps')
        total_reward = 0
        steps = 0
        while not terminal_reached and steps < max_steps:
            # take action based on observation
            nn_output = net.activate(state)
            action = np.argmax(nn_output)

            # perform next step
            state, reward, done, info = env.step(action)
            total_reward += reward
            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                state, reward, done, info = env.step(action)
                total_reward += reward

            steps += 1
            if done:
                terminal_reached = True

        return total_reward


def test_best_agent(generation_count, genome, net, display):
    logger.debug("Generating best agent result: %d", generation_count)
    t_start = datetime.now()

    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('initialisation', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0

    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached:
            if display:
                env.render()

            output = net.activate(state)
            action = np.argmax(output)
            next_state, reward, done, info = env.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env.step(action)
                reward += reward2

            total_rewards += reward
            state = next_state

            steps += 1
            if done:
                terminal_reached = True
        total_steps += steps

    average_steps_per_episode = total_steps / test_episodes
    average_rewards_per_episode = total_rewards / test_episodes

    # save this to file along with the generation number
    entry = [generation_count, average_steps_per_episode, average_rewards_per_episode]
    with open(r'agent_evaluation-{0}.csv'.format(time), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('display', nargs='?', default='true', help='Show display of game. true or false')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    logging.basicConfig(filename='log/debug-{0}.log'.format(time),
                        level=logging.DEBUG, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # Load the properties file
    local_dir = os.path.dirname(__file__)
    logger.debug("Loading Properties")
    props = configparser.ConfigParser()
    prop_path = os.path.join(local_dir, 'properties/{0}/neatem_properties.ini'.format(env.spec.id))
    props.read(prop_path)
    logger.debug("Finished: Loading Properties")

    # Load the config file, which is assumed to live in properties/[environment id] directory
    logger.debug("Loading NEAT Config file")
    config_path = os.path.join(local_dir, 'properties/{0}/config'.format(env.spec.id))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    agent = Neat(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    display_game = True if args.display == 'true' else False
    try:
        agent.execute_algorithm(props.getint('train', 'generation'))

        visualize.plot_stats(agent.stats, ylog=False, view=False, filename="fitness-{0}.svg".format(time))

        # generate test results and record gameplay
        if display_game:
            outdir = 'videos/tmp/neat-data/{0}-{1}'.format(env.spec.id, str(datetime.now()))
            env = wrappers.Monitor(env, directory=outdir, force=True)

        for (generation_count, genome, net) in agent.best_agents:
            test_best_agent(generation_count, genome, net, display_game)

        mfs = sum(agent.stats.get_fitness_mean()[-20:]) / 20.0
        logger.debug("Average mean fitness over last 20 generations: %f", mfs)

        mfs = sum(agent.stats.get_fitness_stat(min)[-20:]) / 20.0
        logger.debug("Average min fitness over last 20 generations: %f", mfs)

        # Use the ten best genomes seen so far as an ensemble-ish control system.
        best_genomes = agent.stats.best_unique_genomes(10)

        # Save the winners
        for n, g in enumerate(best_genomes):
            name = 'winners/winner-{0}'.format(n)
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

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
