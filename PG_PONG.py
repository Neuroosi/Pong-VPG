from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow import keras
import random
import numpy as np
from collections import deque
import gym
import math
from collections import deque
from graphs import graph

##Hyperparameters
learning_rate_policy = 0.0001
learning_rate_value = 0.01
GAMMA = 0.99
EPISODES = 5000
BATCH_SIZE = 5
D = 80
class VALUEFUNCTION_ESTIMATOR:
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        initializer = keras.initializers.HeNormal()
        model.add(Dense(64,input_shape = (D**2,), activation = 'relu' ,kernel_initializer=initializer))
        #model.add(Dense(256, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        initializer2 = keras.initializers.GlorotNormal()
        model.add(Dense(1, activation = 'linear',kernel_initializer=initializer2))
        model.compile(loss='mse',optimizer=adam_v2.Adam(lr=learning_rate_value))
        return model

    def train(self, G, states):
        print("VALUEFUNC_ESTIMATOR LOSS: ", self.model.train_on_batch(np.array(states), np.array(G)))

    def getPrediction(self, states):
        return self.model.predict_on_batch(states)
    

class PLNET:
    def __init__(self):
        self.actionSpaceSize = 1
        self.model = self.create_model()
        self.states = []
        self.rewards = []
        self.actions = []

    def create_model(self):
        model = Sequential()
        initializer = keras.initializers.HeNormal()
        model.add(Dense(200,input_shape = (D**2,), activation = 'relu' ,kernel_initializer=initializer))
        #model.add(Dense(10, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        initializer2 = keras.initializers.GlorotNormal()
        model.add(Dense(self.actionSpaceSize, activation = 'sigmoid',kernel_initializer=initializer2))
        model.compile(loss= 'binary_crossentropy',optimizer=adam_v2.Adam(lr=learning_rate_policy))
        return model


    def update_sample(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        cache = np.zeros(self.actionSpaceSize)
        cache[0] = action
        self.actions.append(cache)

    def discounted_reward(self):
        G = np.zeros(len(self.rewards))
        ##Calculate discounted reward
        cache = 0
        for t in reversed(range(0, len(self.rewards))):
            if self.rewards[t] != 0: cache = 0
            cache = cache*GAMMA + self.rewards[t]
            G[t] = cache
        ##Normalize
        G = (G-np.mean(G))/(np.std(G))
        return G
    def train(self, G):
        ##Train the model
        #self.model.fit(np.array(self.states), np.array(self.actions), epochs = 1, verbose = 1)
        his = self.model.train_on_batch(np.array(self.states), np.array(self.actions), sample_weight=G)
        print(his)
        self.actions, self.states, self.rewards = [],[],[]        
    def saveModel(self):
        self.model.save("bestmodel")
        print("Model saved!")
    def loadModel(self):
        self.model = keras.models.load_model("bestmodel")
        print("Model loaded successfully!")

    def getPrediction(self, state):
        prob = self.model.predict(np.array([state]))
        prob = np.squeeze(prob)
        probs = np.array([1-prob, prob])
        return np.random.choice(2, p = probs)


def getFrame(p):
    p = p[35:195]
    p = p[::2, ::2, 0]
    p[p == 144] = 0
    p[p == 109] = 0
    p[p != 0] = 1
    return p

def makeState(state):
    cache = state[1]-state[0]
    return cache.flatten()

if __name__ == "__main__":
    env = gym.make("PongDeterministic-v4")
    rewards = []
    avgrewards = []
    state = deque(maxlen = 2)
    agent = PLNET()
    value_estimator = VALUEFUNCTION_ESTIMATOR()
    ans = input("Use a pretrained model y/n? ")
    if ans == "y":
        agent.loadModel()
    total_time = 0
    cumureward = 0
    for episode in range(1,EPISODES+500000000000):
        observation = env.reset()
        state.append(getFrame(observation))
        state.append(getFrame(observation))
        #state.append(getFrame(p))
        #state.append(getFrame(p))
        gamereward = 0
        while True:
            action = agent.getPrediction(makeState(state))
            if action == 1:
                observation, reward, done, info = env.step(2)##UP
            else:
                observation, reward, done, info = env.step(3)##DOWN
            agent.update_sample(makeState(state), reward, action)
            state.append(getFrame(observation))
            cumureward += reward
            total_time += 1
            gamereward += reward
            env.render()
            if done:
                print("Running reward: ", gamereward)
                rewards.append(gamereward)
                break
        if episode % 1000 == 0:
            agent.saveModel()
            graph(avgrewards, rewards,"fetajuusto/VPG-PONG")
        if episode % BATCH_SIZE == 0:
            print("Avg batch reward: ", cumureward/BATCH_SIZE, " Episode: ", episode/BATCH_SIZE, " Steps: ", total_time)
            G = agent.discounted_reward()
            states = np.array(agent.states)
            V_ESTIMATES = np.array(value_estimator.getPrediction(states))
            V_ESTIMATES = np.squeeze(V_ESTIMATES)
            V_ESTIMATES = (V_ESTIMATES - np.mean(V_ESTIMATES))/np.std(V_ESTIMATES)
            agent.train(G-V_ESTIMATES)
            value_estimator.train(G, states)
            avgrewards.append(cumureward/BATCH_SIZE)
            cumureward = 0