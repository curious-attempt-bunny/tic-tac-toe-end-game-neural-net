import network
import csv
import numpy as np
import random

class EndGame(object):
  def __init__(self, sizes = [9*1,2]):
    self.net = network.Network(sizes)

  def load(self):
    reader = csv.reader(open('../data/uci-dataset.csv', 'rb'))
    dataset = []
    # translation = { 'x': [1,0,0], 'o': [0,1,0], 'b': [0,0,1], 'negative':0, 'positive': 1}
    translation = { 'x': [1], 'o': [-1], 'b': [0], 'negative':0, 'positive': 1}
    positiveCount = 0
    negativeCount = 0
    for row in reader:
      values = [translation[v] for v in row]
      values = sum(values[0:-1], []) + values[-1:]
      if values[-1] == 0:
        negativeCount += 1
      else:
        positiveCount += 1
      # print values
      dataset.append([np.reshape(values[0:-1], (9*1,1)), values[-1]])

    print "Positive %d, negative %d" % (positiveCount, negativeCount)

    random.seed(10)
    random.shuffle(dataset)

    training = dataset[0:int(0.8*len(dataset))]
    validation = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    test = dataset[int(0.9*len(dataset)):]

    training = [[x, network.vectorized_result(y)] for x,y in training]
    validation = [[x, np.reshape(y, (1,1))] for x,y in validation]
    test = [[x, np.reshape(y, (1,1))] for x,y in test]

    # print validation[0]

    return (training,validation,test)

  def train(self, epochs=100, learning_rate=0.05, lmbda=10):
    training, validation, test = self.load()
    self.net.SGD(training, epochs, 1, learning_rate, lmbda = lmbda,
            evaluation_data=validation,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


