import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


# Question 1: Implementation
def step(state, action):
    new_state = copy.deepcopy(state)
    if action == 'Hit':
        # players turn
        deal = np.random.choice(10) + 1  # create black
        if np.random.choice(3) == 0:  # change to red
            deal = -deal

        new_state[1] += deal

        if new_state[1] > 21 or new_state[1] < 1:
            return new_state, -1  # dealer wins
        else:
            return new_state, 0

    elif action == 'Stick':
        # Dealer turn
        dealer = new_state[0]

        deal = np.random.choice(10) + 1  # create black
        if np.random.choice(3) == 0:  # change to red
            deal = -deal
        dealer += deal

        while dealer < 17 and dealer > 1:
            # Dealer hits
            deal = np.random.choice(10) + 1  # create black
            if np.random.choice(3) == 0:  # change to red
                deal = -deal
            dealer += deal

        if dealer > 21 or dealer < 1:  # player wins
            return new_state, 1

        # dealer sticks
        if dealer < new_state[1]:  # player wins
            return new_state, 1
        elif dealer > new_state[1]:  # dealer wins
            return new_state, -1
        else:  # draw
            return new_state, 0


# Initialize
dealers_first_hand = np.random.choice(10) + 1  # create black
players_first_hand = np.random.choice(10) + 1  # create black

new_state = [dealers_first_hand, players_first_hand]
new_action = 'Hit'
game_status = True

while game_status:
    print("Player: %d || dealer: %d" % (new_state[1], new_state[0]))
    if new_state[1] >= 17:
        new_action = 'Stick'
    new_state, reward = step(new_state, new_action)
    if reward != 0 or new_state[0] != dealers_first_hand:
        game_status = False

print('Reward is: %d' % reward)
print('Final state is: player: %d || dealer: %d' % (new_state[1], new_state[0]))


# Question 2: Monte Carlo

def do_action(state, Ns, Qs, n_constant):
    total_visits = np.sum(Ns[state[0] - 1, state[1] - 1, :])
    epsilon = n_constant / (n_constant + total_visits)

    if np.random.uniform(0, 1) < epsilon:  # do random
        if np.random.choice(2) == 0:
            return 1, 'Stick'
        else:
            return 0, 'Hit'
    else:
        best_action = np.argmax(Qs[state[0] - 1, state[1] - 1, :])
        if best_action == 0:
            return 0, 'Hit'
        else:
            return 1, 'Stick'


n0 = 100
iterations = 20000
dealer_value = 10  # 1:10
player_value = 21  # 1:21
action_value = 2  # Hit or Stick

# Number of visits
N = np.zeros((dealer_value, player_value, action_value))

# Action-Value
Q_mc = np.zeros((dealer_value, player_value, action_value))

# Value Function
V_mc = np.zeros((dealer_value, player_value))

# Lets Train
for episode in range(iterations):
    time_steps = []

    # Initialize
    dealers_first_hand = np.random.choice(10) + 1  # create black
    players_first_hand = np.random.choice(10) + 1  # create black

    new_state = [dealers_first_hand, players_first_hand]
    game_status = True

    while game_status:
        # Choose action
        new_action_int, new_action = do_action(new_state, N, Q_mc, n0)

        # Add to time steps
        time_steps.append((new_state, new_action_int))

        # update number of visits
        N[new_state[0] - 1, new_state[1] - 1, new_action_int] += 1

        # take a step forward
        new_state, reward = step(new_state, new_action)

        if reward != 0 or new_state[0] != dealers_first_hand:
            game_status = False

    # update action-value
    for s, a in time_steps:
        alpha = 1 / N[s[0] - 1, s[1] - 1, a]  # step size
        Q_mc[s[0] - 1, s[1] - 1, a] += alpha * (reward - Q_mc[s[0] - 1, s[1] - 1, a])

# update value function
for dealer_index in range(dealer_value):
    for player_index in range(player_value):
        V_mc[dealer_index, player_index] = np.max(Q_mc[dealer_index, player_index, :])

# plot
x = np.arange(0, dealer_value, 1)
y = np.arange(0, player_value, 1)

x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, V_mc.T, cmap=cm.coolwarm, linewidth=1, antialiased=False)
ax.set_xlabel('Dealer', size=16)
ax.set_ylabel('Player', size=16)
ax.set_zlabel('Optimal Value', size=16)
plt.show()

# Question 3: Sarsa lambda
n0 = 100
iterations = 1000
dealer_value = 10  # 1:10
player_value = 21  # 1:21
action_value = 2  # Hit or Stick

# lambdas
lambda_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

MSE = []
# Lets Train
for lambda_value in lambda_list:
    # Number of visits
    N = np.zeros((dealer_value, player_value, action_value))

    # Action-Value
    Q_sarsa = np.zeros((dealer_value, player_value, action_value))

    # Value Function
    V_sarsa = np.zeros((dealer_value, player_value))

    episode_MSE = []

    for episode in range(iterations):
        # Initialize
        dealers_first_hand = np.random.choice(10) + 1  # create black
        players_first_hand = np.random.choice(10) + 1  # create black

        new_state = [dealers_first_hand, players_first_hand]
        new_action_int, new_action = do_action(new_state, N, Q_sarsa, n0)

        next_action_int = new_action_int
        next_action = new_action

        game_status = True

        # Lambda effect
        E = np.zeros((dealer_value, player_value, action_value))

        while game_status:
            # update visits
            N[new_state[0] - 1, new_state[1] - 1, new_action_int] += 1

            next_state, reward = step(new_state, new_action)

            current_q = Q_sarsa[new_state[0] - 1, new_state[1] - 1, new_action_int]

            if reward != 0 or next_state[0] != dealers_first_hand:
                game_status = False

            if game_status:
                # go greedy
                next_action_int, next_action = do_action(next_state, N, Q_sarsa, n0)
                next_q = Q_sarsa[next_state[0] - 1, next_state[1] - 1, next_action_int]
                delta = reward + next_q - current_q
            else:
                delta = reward - current_q

            # update action_value
            E[new_state[0] - 1, new_state[1] - 1, new_action_int] += 1
            alpha = 1 / N[new_state[0] - 1, new_state[1] - 1, new_action_int]
            Q_sarsa += alpha * delta * E
            E *= lambda_value

            new_state = next_state
            new_action = next_action
            new_action_int = next_action_int

        if lambda_value == 0 or lambda_value == 1:
            episode_MSE.append((np.square(Q_mc - Q_sarsa)).mean(axis=None))

    if lambda_value == 0 or lambda_value == 1:
        fig = plt.figure()
        plt.plot(range(iterations), episode_MSE)
        plt.title(r'$\lambda$ = %.1f' % lambda_value)
        plt.xlabel('Episodes', size=16)
        plt.ylabel('MSE', size=16)
        plt.show()

    # update value function
    for dealer_index in range(dealer_value):
        for player_index in range(player_value):
            V_sarsa[dealer_index, player_index] = np.max(Q_sarsa[dealer_index, player_index, :])

    MSE.append((np.square(Q_sarsa - Q_mc)).mean(axis=None))

fig = plt.figure()
plt.plot(lambda_list, MSE)
plt.xlabel(r'$\lambda$', size=16)
plt.ylabel('MSE', size=16)
plt.show()
