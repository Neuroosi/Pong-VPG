from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow import keras
import tensorflow.keras.backend as K 
import random
import numpy as np
from collections import deque
from ple.games.pong import Pong
from ple import PLE
import math

##Hyperparameters
learning_rate = 0.001
GAMMA = 0.99
EPISODES = 5000
BETA = 0.01

def loss_func(y_true, y_pred):
    cce = keras.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred)+BETA*K.sum(-y_pred*K.log(y_pred))

class VALUEFUNCTION_ESTIMATOR:
    def __init__(self, obsSpaceSize):
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(Dense(100,input_shape = (self.obsSpaceSize,), activation = 'relu' ))
        #model.add(Dense(256, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        model.add(Dense(1, activation = 'linear'))
        model.compile(loss='mse',optimizer=adam_v2.Adam(lr=learning_rate))
        return model

    def train(self, G, states):
        print("VALUEFUNC_ESTIMATOR LOSS: ", self.model.train_on_batch(np.array(states), np.array(G)))

    def getPrediction(self, states):
        return self.model.predict_on_batch(states)
    

class PLNET:
    def __init__(self,  actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
        self.entropy = []
        self.states = []
        self.rewards = []
        self.actions = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(100,input_shape = (self.obsSpaceSize,), activation = 'relu' ))
        #model.add(Dense(10, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        model.add(Dense(self.actionSpaceSize, activation = 'softmax'))
        model.compile(loss= loss_func,optimizer=adam_v2.Adam(lr=learning_rate))
        return model


    def update_sample(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        cache = np.zeros(self.actionSpaceSize)
        cache[action] = 1
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
        
    def calculate_entropy(self, probs):
        return np.sum(-probs*np.log(probs))

    def getPrediction(self, state):
        probs = self.model.predict(np.array([state]))
        probs = np.squeeze(probs)
        self.entropy.append(np.sum(-probs*np.log(probs)))##entropy
        return np.random.choice(self.actionSpaceSize, p = probs)

if __name__ == "__main__":
    #game = FlappyBird(width=288, height=512, pipe_gap=100)
    game = Pong(500,500)
    ans = input("Displayscreen y/n? ")
    displayscreen = False
    if ans == "y":
        displayscreen = True
    p = PLE(game, fps = 30, frame_skip = 3, display_screen=displayscreen)
    p.init()
    state = []
    agent = PLNET(len(p.getActionSet()), len(p.getGameState()))
    value_estimator = VALUEFUNCTION_ESTIMATOR(len(p.getGameState()))
    ans = input("Use a pretrained model y/n ?")
    if ans == "y":
        agent.loadModel()
    avg_reward = 0
    total_time = 0
    max_t = -10000000
    max_reward = -10000000
    rewards = []
    times = []
    for episode in range(1,EPISODES+500000000000):
        p.reset_game()
        cumureward = 0
        state = np.array(list(p.getGameState().values()))
        t = 0
        while True:
            state = (state - np.mean(state))/np.std(state)
            action = agent.getPrediction(state)
            reward = p.act(p.getActionSet()[action])
            next_state = np.array(list(p.getGameState().values()))
            agent.update_sample(state, reward, action)
            state = next_state
            t+=1
            cumureward += reward
            avg_reward += reward
            if p.game_over() == True:
                break
        if episode % 1000 == 0:
            agent.saveModel()
        max_t = max(max_t, t)
        total_time += t
        max_reward = max(cumureward, max_reward)
        rewards.append(cumureward)
        times.append(t)
        print("Beta: ", BETA," Score: ",cumureward, " Avg score: ",avg_reward/episode, " Max score: " ,max_reward,"Time:", t," Max time: ", max_t, " Avg time: " ,total_time/episode,  " Episode:", episode, " Total time", total_time)
        if episode > 100:
            cache = np.array(rewards)
            print("Moving average of rewards last 100 episodes: ", np.convolve(cache[len(cache)-100:], np.ones(100), mode='valid')/100)
            cache = np.array(times)
            print("Moving average of time spent per episode last 100 episodes: ", np.convolve(cache[len(cache)-100:], np.ones(100), mode='valid')/100)
        G = agent.discounted_reward()
        states = np.array(agent.states)
        V_ESTIMATES = np.array(value_estimator.getPrediction(states))
        V_ESTIMATES = np.squeeze(V_ESTIMATES)
        agent.train(G-V_ESTIMATES)
        value_estimator.train(G, states)