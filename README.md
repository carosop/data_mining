# How to predict who is playing a game of Starcraft 2
We consider the video game StarCraft 2. This real-time strategy game is played competitively by cyber-athletes as an electronic sport (esport), and most of the competitions are played online. A major issue for competition organizers is that they need to verify the identify of a player, for being sure that it is not another player (a friend with better chances, …).

For that matter, we propose to study how it is possible to determine who is playing given a behavioral trace (game events produced by the player) by designing a prediction model using machine learning techniques.

We were provided with a large set of SC2 replay data, from which we needed to build features and train a model.


We used different machine learning techniques as DecisionTree, RandomForestClassifier with tuned hyperparameters, and combination of RandomForestClassifier and AdaBoost in order to obtain a better score.

We try to explain what we have done step by step through this 3 Jupyter Notebooks:

## Notebooks

- [Different techniques of Model Training, Testing and Data Visualisation](StarCraft_Player_Prediction.ipynb)
- [Feature Importance Analysis](Feature_Importance.ipynb)
- [Feature Reduction with best prediction accuracy](train_jupyter.ipynb)



