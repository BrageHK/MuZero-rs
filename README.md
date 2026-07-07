# MuZero-rs - An implementaiton of MuZero in Rust

# Prerequisites

## sdl2_gfx

Arch:
```bash
sudo pacman -S sdl2_gfx
```

Mac:
```bash
brew install sdl2_gfx
```

## AMD path variables

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

# Result

![Cartpole](media/cartpole.gif)

# TODO:

[] Make NN configurable
[] Backend generic training
[] Reanalyze
[] TicTacToe
[] Othello
[] WASM