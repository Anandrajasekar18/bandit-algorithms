from bandit_algo import create_testbed
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(1)

n_plays = 2000
n_steps = 1000
n_arms = 10


epsilon = [0, 0.01, 0.1, 1]
plt.rc('text',usetex=True)
plt.rc('font', family='serif')


fig, ax = plt.subplots(2, 1)
for eps in epsilon:

    ### Call testbed class and implement epsilon greedy policy for various values of epsilon
    testbed = create_testbed(n_arms,n_steps,n_plays)
    testbed.epsilon_greedy(eps)

    
    ax[0].plot(np.arange(n_steps)+1, testbed.avg_reward, label = r'$\epsilon$ : ' + str(eps) )
    ax[1].plot(np.arange(n_steps)+1, testbed.optim_arm, label = r'$\epsilon$ : ' + str(eps) )

ax[0].set_title(r'$\epsilon$-greedy : Average Reward Vs Steps({} arms)'.format(testbed.n_arms))
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Average reward')
ax[0].legend()

ax[1].set_title(r'$\epsilon$-greedy : $\%$ Optimal arm Vs Steps({} arms)'.format(testbed.n_arms))
ax[1].set_xlabel('Steps')
ax[1].set_ylabel(r'$\%$Optimal arm')
ax[1].legend()

fig.tight_layout()
plt.show()



