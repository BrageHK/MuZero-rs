# MuZero-rs - Optimized and parallel MuZero in Rust

This project is based on the [MuZero](https://arxiv.org/abs/1911.08265) paper by DeepMind.
The main problem of Reinforcement Learning (RL) in many cases is skill issue. RL algorithms
can have astronomical training speedup by not using naive Python implmenetations. That is why
this program is written in Rust.

# Prerequisites

## Packages

This is requried for CartPole and the makefile to work.

Arch:
```bash
sudo pacman -S sdl2_gfx make yq
```

Mac:
```bash
brew install sdl2_gfx pkgconf make yq
```


## AMD path variables

Only needed if using rocm backend with GFX version 11.0.0

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## Training

TGo to [config.yaml](configs/config.yaml) and choose a GPU compute backend
that is compatible with your system. Then use `make` to run.

```bash
make train
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