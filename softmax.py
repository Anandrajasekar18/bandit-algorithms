from bandit_algo import create_testbed
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(1)

n_plays = 2000
n_steps = 1000
n_arms = 10


temp = [0.01,0.1, 1, 10]
plt.rc('text',usetex=True)
plt.rc('font', family='serif')


fig, ax = plt.subplots(2, 1)
for t in temp:

    ### Call testbed class and implement softmax policy for various values of temperature
    testbed = create_testbed(n_arms,n_steps,n_plays)
    testbed.softmax(t)

    
    ax[0].plot(np.arange(n_steps)+1, testbed.avg_reward, label = r'temp : ' + str(t) )
    ax[1].plot(np.arange(n_steps)+1, testbed.optim_arm, label = r'temp : ' + str(t) )

ax[0].set_title(r'Softmax : Average Reward Vs Steps({} arms)'.format(testbed.n_arms))
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Average reward')
ax[0].legend()

ax[1].set_title(r'Softmax : $\%$Optimal arm Vs Steps({} arms)'.format(testbed.n_arms))
ax[1].set_xlabel('Steps')
ax[1].set_ylabel(r'$\%$Optimal arm')
ax[1].legend()

fig.tight_layout()
plt.show()