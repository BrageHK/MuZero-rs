# MuZero-rs - Optimized and parallel MuZero in Rust

This project is based on the [MuZero](https://arxiv.org/abs/1911.08265) paper by DeepMind.
The main problem of Reinforcement Learning (RL) in many cases is skill issue. RL algorithms
can have astronomical training speedup by not using naive Python implmenetations, and instead
using compiled code though either Jax or a fast programming language. That is why this program 
exists and is written in Rust.

# Table of contents


# Algirthm information

This project is based on original [MuZero](https://arxiv.org/abs/1911.08265) paper with
the MuZero Reanalzye algorithm. A number of improvements of this algorithm have been made 
by great academics, and this project is a combination of multiple of these improvements.
The following table shows what improvements are made over the base MuZero algorithm and 
which paper it is from:

| Improvement    | Paper |
| :----------- | :---------- |
| SimSam Consistency Loss  | [EfficientZeroV2](https://arxiv.org/abs/2403.00564) |
| Global Pooling | [KataGo](https://arxiv.org/abs/1902.10565) |
| Gumbel MuZero | [Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) |


# Getting started

Follow each of these steps carefully and make sure you have
all the requried packages. If you are not interested in training 
your own model, you can instead [start the othello website locally]()

## Packages and programming language

* **Rust**: [Programming language](https://rust-lang.org/tools/install/)
* **sdl2_gfx**: CartPole game environment
* **make**: Makefile to easily get started
* **yq**: Used in Makefile

Arch:
```bash
sudo pacman -S sdl2_gfx make yq
```

Mac:
```bash
brew install sdl2_gfx pkgconf make yq protobuf
```

## Training

TGo to [config.yaml](configs/config.yaml) and choose a GPU compute backend
that is compatible with your system. Then use `make` to run.

```bash
make train
```

## Distributed training across multiple PCs

Training can be split across several machines on the same local network,
communicating over gRPC:

- **Coordinator** (`distributed.role: Coordinator`) — one machine. Owns the
  replay buffer and the canonical weights, runs the optimizer step, and
  computes gradients on its own sampled batch alongside any trainer workers
  (synchronous data-parallel training).
- **Trainer worker** (`distributed.role: TrainerWorker`) — zero or more
  machines. Each round it fetches a fresh batch and the current weights from
  the coordinator, computes a gradient, and sends it back. It never applies an
  optimizer step itself, so replicas can't drift even across different GPUs.
- **Self-play worker** (`distributed.role: SelfPlayWorker`) — as many machines
  as you have. Runs self-play and streams finished games to the coordinator,
  receiving a fresh network every `inference_update_interval` training steps.

Every node needs the same `network_type`/`environment`/`linear`/`resnet`/
`projection` section (the network architecture must match exactly), but
`training_backend`/`inference_backend` can be picked per machine to match its
hardware. Each role reads its own config file under
[configs/distributed](configs/distributed) — `coordinator.yaml`,
`trainer_worker.yaml`, `selfplay_worker.yaml` — so fill in the one matching
each machine's role (via `make config`, or by copying its `.example` file),
setting `distributed.coordinator_addr` to the coordinator's LAN address.

```bash
# on the coordinator machine
make coordinator

# on each trainer worker
make trainer-worker

# on each self-play machine
make selfplay-worker
```

# Othello in the browser

You can watch bots play Othello against each other, or play against bots yourself
in the browser with WebAssembly on the Flex backend with CPU SIMD instructions.

## Prerequisites

Install the WASM stuff:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Export trained model checkpoint to WASM binary

If you have trained a Othello MuZero agent, you can used it with WASM
by exporting it with this command:

```bash
# optional arg: checkpoint path without extension
cargo run --bin export_web            

# 2. Build the wasm package
make web-build

# 3. Serve it
make web-serve
```

# Results

MuZero can play many different games. Here are som examples of how the agent
learns to play.

## Cartpole

After running training for a few minutes on a M2 Pro mac, the agent learns to play CartPole perfectly.

![Cartpole](media/cartpole.gif)