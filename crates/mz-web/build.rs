//! Converts `onnx/chess_mamba.onnx` into a Burn module + embedded weights,
//! generated into `$OUT_DIR/chess_mamba/chess_mamba.rs` and included by
//! `src/model/chess_mamba.rs`.
//!
//! The onnx file itself comes from a separate repo (bee-chess's
//! `training/src/bee_training/chess_mamba/export_onnx.py --static-batch`,
//! run against `training/checkpoints/ThisTimeForSure/best.pt` -- the best
//! checkpoint, by val loss, of that run, not `latest.pt`) -- there's no
//! PyTorch/Python toolchain here to regenerate it from scratch, so it's
//! committed as a source input, the same way Othello/Chess's `assets/*.bin`
//! are committed build *outputs* from `export_web`.

use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    println!("cargo:rerun-if-changed=onnx/chess_mamba.onnx");
    ModelGen::new()
        .input("onnx/chess_mamba.onnx")
        .out_dir("chess_mamba/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
