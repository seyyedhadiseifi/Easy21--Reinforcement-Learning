# Easy21--Reinforcement-Learning
* solutions to first three question of [David Silver's RL course project Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf)

# Part A: Implementation of easy21
In first part of the code, we have implemented the step function which get a state and an action and plays the game by rules. One sample output where both player and dealer hit while their sum is below 17:
Player: 10 || dealer: 7
Player: 18 || dealer: 7
Reward is: 1

# Part B: Monte-Carlo Control in Easy21
In this part, we are using the function step defined in previous section along with assumption stated in the handout to apply the Monte-Carlo control to easy21.

# Part C: TD Learning in Easy21
In this part, we will implement sarsa(λ) for easy21. We have kept the value-action function of the Monte-Carlo control to calculate the mean squared error over all states and actions, comparing the Monte-Carlo and sarsa(λ). 

