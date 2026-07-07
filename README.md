# MuZero-rs - Optimized and parallel MuZero in Rust

This project is based on the [MuZero](https://arxiv.org/abs/1911.08265) paper by DeepMind.
The main problem of Reinforcement Learning (RL) in many cases is skill issue. RL algorithms
can have astronomical training speedup by not using naive Python implmenetations. That is why
this program is written in Rust.

# Result

After running the parallel training for a few minutes on a M2 Pro mac, the agent learns to play CartPole perfectly.

![Cartpole](media/cartpole.gif)

# TODO:

* [ ] Reanalyze
* [ ] TicTacToe
* [ ] Othello
* [ ] WASM - play against muzero in othello