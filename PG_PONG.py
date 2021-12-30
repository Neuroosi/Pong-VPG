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
learning_rate = 0.001
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
        model.add(Dense(100,input_shape = (self.obsSpaceSize,), activation = 'relu' ))
        #model.add(Dense(256, activation = 'relu'))
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
        self.model.save("/home/joel/PONG-PG/bestmodel")
        print("Model saved!")
    def loadModel(self):
        self.model = keras.models.load_model("/home/joel/PONG-PG/bestmodel/")
        

    def getPrediction(self, state):
        probs = self.model.predict(np.array([state]))
        probs = np.squeeze(probs)
        return np.random.choice(self.actionSpaceSize, p = probs)

if __name__ == "__main__":
    #game = FlappyBird(width=288, height=512, pipe_gap=100)
    game = Pong(500,500)
    p = PLE(game, fps = 30, frame_skip = 3, display_screen=True)
    p.init()
    state = []
    agent = PLNET(len(p.getActionSet()), len(p.getGameState()))
    ans = input("Use a pretrained model y/n?")
    if ans is "Y":
        agent.loadModel()
    avg_reward = 0
    total_time = 0
    max_t = -10000000
    max_reward = -10000000
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
            if t % 10000 == 0:
                agent.saveModel()
            if p.game_over() == True:
                break
        estimate_value_function = agent.discounted_reward()
        max_t = max(max_t, t)
        total_time += t
        max_reward = max(cumureward, max_reward)
        print("Score: ",cumureward, " Avg score: ",avg_reward/episode, " Max score: " ,max_reward,"Time:", t," Max time: ", max_t, "Avg time: " ,total_time/episode, "Discounted reward max:", np.amax(estimate_value_function), " Min discounted reward: ", np.amin(estimate_value_function) ,"Discounted reward avg: ", np.mean(estimate_value_function) ," Episode:", episode)
        agent.train(estimate_value_function)