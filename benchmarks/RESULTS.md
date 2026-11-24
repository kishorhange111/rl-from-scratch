| env | method | budget | greedy eval (mean ± std) | IQM | train AUC | median steps to solve | seeds solved | wall/seed |
|---|---|---|---|---|---|---|---|---|
| Acrobot-v1 | dqn | 100,000 | -75.8 ± 5.4 | -76.2 | -132.8 | 20,815 | 5/5 | 512s |
| Acrobot-v1 | ppo | 200,000 | -109.8 ± 54.9 | -87.8 | -259.9 | 200,000 | 0/5 | 78s |
| CartPole-v1 | dqn | 50,000 | 456.6 ± 97.0 | 500.0 | 184.1 | 50,000 | 1/5 | 213s |
| CartPole-v1 | dqn_vanilla | 50,000 | 388.2 ± 153.2 | 409.7 | 225.3 | 45,068 | 3/5 | 205s |
| CartPole-v1 | ppo | 100,000 | 442.1 ± 40.9 | 439.7 | 175.6 | 100,000 | 0/5 | 30s |
| CartPole-v1 | ppo_noclip | 100,000 | 482.9 ± 17.3 | 485.0 | 166.4 | 100,000 | 0/5 | 29s |
| CartPole-v1 | ppo_zoo | 100,000 | 500.0 ± 0.0 | 500.0 | 341.7 | 51,584 | 5/5 | 101s |
| CartPole-v1 | reinforce | 100,000 | 483.3 ± 21.3 | 490.1 | 372.1 | 56,713 | 4/5 | 58s |
| CartPole-v1 | reinforce_nobaseline | 100,000 | 500.0 ± 0.0 | 500.0 | 365.6 | 55,246 | 5/5 | 57s |
