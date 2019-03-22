import numpy as np 
import matplotlib.pyplot as plt
import time
import random

class create_testbed:
    '''Creates a testbed for implementing various bandit algorithms with different number of arms, steps and plays.'''
    def __init__(self,n_arms, n_steps, n_plays):
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.n_plays = n_plays
        self.qstar = np.random.randn(self.n_plays,self.n_arms) #True mean over n_plays for every arm
        self.best_arm = np.argmax(self.qstar,axis = 1) #Best mean over n_plays
        self.Q = np.zeros((self.n_plays,self.n_arms))  #Estimated mean over n_plays for every arm
        self.n_taken = np.zeros((self.n_plays,self.n_arms)) #Number of times every arm is taken over n_plays
        self.avg_reward = []
        self.optim_arm = []
        self.step = 0




    def epsilon_greedy(self,epsilon):
        '''Choose arm with highest estimated reward with 1-epsilon porbability and randomly with epsilon probability'''
        for i in range(self.n_steps):
            avg_rew = 0 
            optim = 0
            for j in range(self.n_plays):
                prob = random.uniform(0,1)  

                #### Choose the arm ####
                if (prob > epsilon):
                    arm_choose = np.argmax(self.Q[j,:])
                else :
                    arm_choose = np.random.choice(self.n_arms)

                ### Get reward
                new_rew = np.random.normal(loc = self.qstar[j,arm_choose])

                ###Update arm parameters
                self.Q[j,arm_choose] = self.Q[j,arm_choose] + 1/(self.n_taken[j,arm_choose]+1)*(new_rew - self.Q[j,arm_choose])
                self.n_taken[j,arm_choose]+=1

                ###Update average parameters
                avg_rew+=new_rew
                if arm_choose == self.best_arm[j]:
                    optim+=1
            self.step+=1
            self.avg_reward.append(avg_rew/self.n_plays)
            self.optim_arm.append(np.mean(optim)*100/self.n_plays)

    def softmax(self,temp):
        '''Sample arm from softmax distribution'''

        for i in range(self.n_steps):
            avg_rew = 0 
            optim = 0
            for j in range(self.n_plays):

                #### Choose the arm###
                prob = np.exp(self.Q[j,:]/temp)/np.sum(np.exp(self.Q[j,:]/temp))
                arm_choose = np.random.choice(self.n_arms, p = prob.squeeze())

                #### Get reward
                new_rew = np.random.normal(loc = self.qstar[j,arm_choose])

                ### Update arm parameters
                self.Q[j,arm_choose] = self.Q[j,arm_choose] + 1/(self.n_taken[j,arm_choose]+1)*(new_rew - self.Q[j,arm_choose])
                self.n_taken[j,arm_choose]+=1

                #### Update average parameters
                avg_rew+=new_rew
                if arm_choose == self.best_arm[j]:
                    optim+=1
            self.step+=1
            self.avg_reward.append(avg_rew/self.n_plays)
            self.optim_arm.append(np.mean(optim)*100/self.n_plays)

    def ucb1(self):
        '''Pick the arm with highest upper confidence bound'''

        ### Play each arm once
        self.Q = np.random.normal(self.qstar)  
        self.n_taken+=1

        self.step+=self.n_arms
        self.avg_reward+= list(np.mean(self.Q,axis=0))
        self.optim_arm+= list(np.bincount(self.best_arm, minlength=self.n_arms)*100/self.n_plays)

        for i in range(self.n_steps-self.n_arms):
            avg_rew = 0 
            optim = 0
            for j in range(self.n_plays):
                ## Calculate UCB and choose the arm
                upp_conf = self.Q[j,:] + np.sqrt(2*np.log(self.step)/self.n_taken[j,:])
                arm_choose = np.argmax(upp_conf)

                ## Get reward
                new_rew = np.random.normal(loc = self.qstar[j,arm_choose])

                ##Update arm parameters
                self.Q[j,arm_choose] = self.Q[j,arm_choose] + 1/(self.n_taken[j,arm_choose]+1)*(new_rew - self.Q[j,arm_choose])
                self.n_taken[j,arm_choose]+=1

                ###Update average parameters
                avg_rew+=new_rew
                if arm_choose == self.best_arm[j]:
                    optim+=1
            
            self.step+=1
            self.avg_reward.append(avg_rew/self.n_plays)
            self.optim_arm.append(np.mean(optim)*100/self.n_plays)
        print (len(self.avg_reward))


if __name__ == '__main__':
    np.random.seed(1)
    tic = time.time()
    n_arms = 10
    n_steps = 1000
    testbed = create_testbed(n_arms,n_steps,20)
    testbed.epsilon_greedy(0.01)
    plt.figure(1)
    plt.plot(np.arange(n_steps)+1, testbed.avg_reward)

    plt.figure(2)
    plt.plot(np.arange(n_steps)+1, testbed.optim_arm)

    plt.show()

    toc = time.time()
    print(toc-tic)


    