from bandit_algo import create_testbed
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(1)

n_plays = 2000
n_steps = 1000
n_arms = 10  ####10 arms


eps = 0.1
temp = 0.1

plt.rc('text',usetex=True)
plt.rc('font', family='serif')




fig, ax = plt.subplots(2, 1)
testbed = create_testbed(n_arms,n_steps,n_plays)
testbed.epsilon_greedy(eps)  ### Implement epsilon greedy
ax[0].plot(np.arange(n_steps)+1, testbed.avg_reward, label = r'$\epsilon$ greedy ($\epsilon$: ' + str(eps) +')' )
ax[1].plot(np.arange(n_steps)+1, testbed.optim_arm, label = r'$\epsilon$ greedy ($\epsilon$: ' + str(eps) +')')

testbed = create_testbed(n_arms,n_steps,n_plays)
testbed.softmax(temp) ### Implement softmax
ax[0].plot(np.arange(n_steps)+1, testbed.avg_reward, label = r'softmax (temp: ' + str(temp) +')' )
ax[1].plot(np.arange(n_steps)+1, testbed.optim_arm, label = r'softmax (temp: ' + str(temp) + ')')

testbed = create_testbed(n_arms,n_steps,n_plays)
testbed.ucb1() #### Implement UCB1
ax[0].plot(np.arange(n_steps)+1, testbed.avg_reward, label = r'ucb1')
ax[1].plot(np.arange(n_steps)+1, testbed.optim_arm, label = r'ucb1')

ax[0].set_title(r'Comparison : Average Reward Vs Steps({} arms)'.format(testbed.n_arms))
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Average reward')
ax[0].legend()

ax[1].set_title(r'Comparison : $\%$ Optimal arm Vs Steps({} arms)'.format(testbed.n_arms))
ax[1].set_xlabel('Steps')
ax[1].set_ylabel(r'$\%$Optimal arm')
ax[1].legend()

fig.tight_layout()
plt.show()