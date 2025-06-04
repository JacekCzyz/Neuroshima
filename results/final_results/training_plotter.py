import matplotlib.pyplot as plt
import csv
import numpy as np

filename = 'reward_time_ppo_small_minmax_500-000_vs_hard.csv'

average_rewards = []
game_results = []

result_map = {
    "1won": 1,
    "2won": -1,
    "tie": 0
}

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    lines = list(reader)

    for i in range(0, len(lines)-1, 2):
        if i + 1 >= len(lines):
            break
        rewards_line = lines[i]
        result_line = lines[i+1][0].strip()

        rewards = [float(r) for r in rewards_line if r != '']
        average_reward = sum(rewards) / len(rewards) if rewards else 0
        average_rewards.append(average_reward)

        result_value = result_map.get(result_line, None)
        if result_value is None:
            print(f"Nieznany wynik gry: {result_line}")
            exit(1)
        game_results.append(result_value)

window_size = 100
moving_averages = []
for i in range(len(average_rewards)):
    window = average_rewards[max(0, i - window_size + 1):i + 1]
    moving_averages.append(sum(window) / len(window))

num_games = len(game_results)
segment_size = num_games // 5

group_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
win_counts = []
loss_counts = []
tie_counts = []

for i in range(5):
    start = i * segment_size
    end = (i + 1) * segment_size if i < 4 else num_games
    segment = game_results[start:end]
    win_counts.append(segment.count(1))
    loss_counts.append(segment.count(-1))
    tie_counts.append(segment.count(0))

plt.figure(figsize=(14, 6))

# Plot1
plt.subplot(1, 2, 1)
plt.plot(moving_averages, linestyle='-', color='blue')
plt.title('Średnia nagród z ostatnich 100 gier')
plt.xlabel('Numer gry')
plt.ylabel('Średnia nagroda')

# Plot2
x = np.arange(len(group_labels))
width = 0.25

plt.subplot(1, 2, 2)
plt.bar(x - width, loss_counts, width=width, label='wygrana przeciwnika', color='red')
plt.bar(x, tie_counts, width=width, label='Remis', color='gray')
plt.bar(x + width, win_counts, width=width, label='wygrana agenta', color='green')
plt.title('Rozkład wyników w trakcie nauki')
plt.xlabel('Zakres procentowy gier')
plt.ylabel('Liczba gier')
plt.xticks(x, group_labels)
plt.legend()

plt.tight_layout()
plt.show()