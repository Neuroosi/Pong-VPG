from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow import keras
import random
import numpy as np
from collections import deque
from ple.games.pong import Pong
from ple import PLE
import math
import skimage
from collections import deque
from graphs import graph

##Hyperparameters
learning_rate = 0.0001
GAMMA = 0.9
EPISODES = 5000
BATCH_SIZE = 5
D = 80
class VALUEFUNCTION_ESTIMATOR:
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        initializer = keras.initializers.HeNormal()
        model.add(Dense(200,input_shape = (D**2,), activation = 'relu' ,kernel_initializer=initializer))
        #model.add(Dense(256, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        initializer2 = keras.initializers.GlorotNormal()
        model.add(Dense(1, activation = 'linear',kernel_initializer=initializer2))
        model.compile(loss='mse',optimizer=adam_v2.Adam(lr=learning_rate))
        return model

    def train(self, G, states):
        print("VALUEFUNC_ESTIMATOR LOSS: ", self.model.train_on_batch(np.array(states), np.array(G)))

    def getPrediction(self, states):
        return self.model.predict_on_batch(states)
    

class PLNET:
    def __init__(self):
        self.actionSpaceSize = 2
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
        model.add(Dense(self.actionSpaceSize, activation = 'softmax',kernel_initializer=initializer2))
        model.compile(loss= 'categorical_crossentropy',optimizer=adam_v2.Adam(lr=learning_rate))
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

    def getPrediction(self, state):
        probs = self.model.predict(np.array([state]))
        probs = np.squeeze(probs)
        return np.random.choice(self.actionSpaceSize, p = probs)


def getFrame(p):
    state = skimage.color.rgb2gray(p.getScreenRGB())
    state = skimage.transform.resize(state, (D,D))
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state[state != 0] = 1
    return state

def makeState(state):
    cache = state[0]-state[1]-state[2]-state[3]
    cache[cache != 0] = 1
    return cache.flatten()

if __name__ == "__main__":
    #game = FlappyBird(width=288, height=512, pipe_gap=100)
    game = Pong(500,500)
    rewards = []
    avgrewards = []
    p = PLE(game, fps = 30, frame_skip = 4, display_screen=True)
    p.init()
    state = deque(maxlen = 4)
    agent = PLNET()
    value_estimator = VALUEFUNCTION_ESTIMATOR()
    ans = input("Use a pretrained model y/n? ")
    if ans == "y":
        agent.loadModel()
    total_time = 0
    cumureward = 0
    for episode in range(1,EPISODES+500000000000):
        p.reset_game()
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        while True:
            action = agent.getPrediction(makeState(state))
            reward = p.act(p.getActionSet()[action])
            agent.update_sample(makeState(state), reward, action)
            state.append(getFrame(p))
            cumureward += reward
            total_time += 1
            if p.game_over() == True:
                break
        if episode % 1000 == 0:
            agent.saveModel()
            graph(rewards,"fetajuusto/VPG-PONG")
        if episode % BATCH_SIZE == 0:
            print("Avg batch reward: ", cumureward/5, " Episode: ", episode/BATCH_SIZE, " Steps: ", total_time)
            G = agent.discounted_reward()
            states = np.array(agent.states)
            V_ESTIMATES = np.array(value_estimator.getPrediction(states))
            V_ESTIMATES = np.squeeze(V_ESTIMATES)
            agent.train(G-V_ESTIMATES)
            value_estimator.train(G, states)
            rewards.append(cumureward/5)
            cumureward = 0