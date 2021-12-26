import matplotlib.pyplot as plt
import numpy as np
import time, random
import os


def cross_loss_curve(total_reward, a_loss, c_loss, step_reward):
    plt.plot(np.array(c_loss), c='b', label='critic_loss', linewidth=0.2)
    plt.plot(np.array(a_loss), c='g', label='actor_loss', linewidth=0.2)
    plt.plot(np.array(step_reward), c='y', label='step_reward', linewidth=0.2)
    plt.plot(np.array(total_reward), c='r', label='total_rewards', linewidth=0.2)
    plt.legend(loc='best')
    # plt.ylim(-15,15)
    # plt.ylim(-0.25,0.05)
    plt.ylabel('loss/100 & reward')
    plt.xlabel('Training Episode')
    plt.grid()
    plt.savefig("history.png")
    plt.close()