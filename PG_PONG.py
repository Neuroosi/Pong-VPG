from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow import keras
import random
import numpy as np
from collections import deque
from ple.games.pong import Pong
from ple import PLE
##Hyperparameters
learning_rate = 0.0001
REPLAY_MEMORY_SIZE=100000
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000

class PLNET:
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.states = []
        self.rewards = []
        self.actions = []
    def create_model(self):
        model = Sequential()
        model.add(Dense(256,input_shape = (self.obsSpaceSize,) ))
        model.add(Dense(256, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        model.add(Dense(self.actionSpaceSize, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=adam_v2.Adam(lr=learning_rate))
        return model

    def update_sample(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        cache = np.zeros(self.actionSpaceSize)
        cache[action] = 1
        self.actions.append(cache)

    def train(self):
        G = []
        ##Calculate discounted reward
        for t in range(len(self.states)):
            v_t = [reward*GAMMA**(k+t-t) for k, reward in enumerate(self.rewards[t:len(self.rewards)])]
            G.append(np.sum(v_t))
        ##Normalize
        G = np.array(G)
        G = (G-np.mean(G))/np.std(G)
        ##Train the model
        self.model.fit(np.array(self.states), np.array(self.actions), sample_weight= G, epochs = 1, verbose = 1)
        self.actions, self.states, self.rewards = [],[],[]        
    def saveModel(self):
        self.model.save("/home/joel/PONG-PG/bestmodel")
    def loadModel(self):
        self.model = keras.models.load_model("/home/joel/PONG-PG/bestmodel/")
        

    def getPrediction(self, state):
        probs = self.model.predict(np.array([state]))
        probs = np.squeeze(probs)
        return np.random.choice(self.actionSpaceSize, p = probs)

if __name__ == "__main__":
    game = Pong(500,500)
    p = PLE(game, fps = 30, display_screen=False)
    p.init()
    state = []
    agent = PLNET(len(p.getActionSet()), len(p.getGameState()))
    t = 0
    for episode in range(EPISODES+500000000000):
        p.reset_game()
        cumureward = 0
        state = np.array(list(p.getGameState().values()))
        while True:
            action = agent.getPrediction(state)
            reward = p.act(p.getActionSet()[action])
            next_state = np.array(list(p.getGameState().values()))
            agent.update_sample(state, reward, action)
            state = next_state
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel()
            if p.game_over() == True:
                break
        print("Score: ",cumureward,"Time:", t, " Episode:", episode)
        agent.train()