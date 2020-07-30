# A Reinforcement Learning [Battlesnake](http://play.battlesnake.com) written in Python.

This battlesnake was trained with Proximal Policy Optimization on Google Colab, using [this article](https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428) by Cory Binnersley and Asymptotic Labs as a guide. The Colab notebook that I used can be found [here](https://colab.research.google.com/drive/1qJ92-eW73LRHxBXa2AMcs-RuNpVihLMV?usp=sharing).

Building off of the work of Cory Binnersley and Arthur Fimino, the [gym](https://github.com/jpulmano/gym-battlesnake) that I utilized was changed slightly to allow custom rewards for the number of kills the snake accumulates, thus allowing me to train a rather aggressive snake.

### Technologies

This Battlesnake uses [Python 3.7](https://www.python.org/), [CherryPy](https://cherrypy.org/), and [Heroku](https://heroku.com). The ML foundation primarily uses [PyTorch](https://pytorch.org/).

### Deploying This Snake

For prerequisites and instructions on how to deploy the snake to Heroku, please see the [original repository](https://github.com/BattlesnakeOfficial/starter-snake-python) by BattlesnakeOfficial.

Feel free to fork this repo to deploy your own ML snake! To get an idea of how this repository operates, here's the flow:

1. `server.py` creates the policy/neural network (found in `my_model.py`) by loading in the weights saved from training
2. `server.py` also creates a game generator object (found in `src/generator.py`), which is used to convert the JSON returned by the game into input for the policy
3. Given the converted input, the policy predicts the best move to take for every turn and returns that move to the Battlesnake game

That's it! There are also some heuristics that I developed which will prevent the snake from randomly hitting walls, other snakes, etc. However, the ML model is typically smart enough to never do those things.

If you want to train your own ML model, I recommend using a copy of the notebook and tweaking the parameters as you see fit; however, the neural network architecture must be consistent between training and how it is loaded in this repository (see `src/my_model.py`)

Once your training has finished, you may save the weights from the model (in `.pt` format) into the `weights` folder of this repository, and load them in `server.py` for deployment.
