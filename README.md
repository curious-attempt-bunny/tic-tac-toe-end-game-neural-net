# Overview

Experimenting with classification of the [UCI dataset for tic-tac-toe endgames](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame).

# Requirements

* python
* numpy

# Usage

```
cd src
python
```

then

```
import endgame
endgame.EndGame().train()
```

experiment with

```
endgame.EndGame(sizes=[9,5,2]).train(epochs=10,learning_rate=0.5,lmbda=1)
```

# Results

Great results!

```
>>> endgame.EndGame().train()
Epoch 0 training complete
Cost on training data: 0.927163226245
Accuracy on training data: 592 / 766
Accuracy on evaluation data: 78 / 96
Epoch 1 training complete
Cost on training data: 0.749859588535
Accuracy on training data: 665 / 766
Accuracy on evaluation data: 87 / 96
Epoch 2 training complete
Cost on training data: 0.625722095227
Accuracy on training data: 702 / 766
Accuracy on evaluation data: 90 / 96
Epoch 3 training complete
Cost on training data: 0.54044731408
Accuracy on training data: 728 / 766
Accuracy on evaluation data: 91 / 96
Epoch 4 training complete
Cost on training data: 0.474378455542
Accuracy on training data: 748 / 766
Accuracy on evaluation data: 95 / 96
Epoch 5 training complete
Cost on training data: 0.422836937739
Accuracy on training data: 752 / 766
Accuracy on evaluation data: 96 / 96
```

# Learnings

* No hidden layer was needed!
* Mini-batch sizes of 1 worked best (i.e. fully online)!
* Encoding x / o / b as [1 / -1 / 0 or 1,0,0 / 0,1,0 / 0,0,1](https://github.com/curious-attempt-bunny/tic-tac-toe-end-game-neural-net/blob/master/src/endgame.py#L13-L14) achieved equivalent results!

# Credits

network.py comes from [mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py).