// Runs the wasm engine off the main thread so a slow Alpha-Beta search (fully
// synchronous Rust, no yield points) or a long MuZero search never blocks the
// page's UI thread — only this worker's own thread stalls while it computes.
import init, { create_chess, chess_bot_move, create_chess_mamba } from "../pkg/mz_web.js";

let chessGame = null;
let chessMambaBot = null;

onmessage = async (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "reset": {
      chessGame.reset();
      break;
    }
    case "mirrorMove": {
      // Keeps chessGame's history-dependent observation in sync with the
      // main thread's chess.js state, regardless of which engine is selected.
      if (!chessGame.play_uci(msg.uci)) {
        console.error(`chessGame rejected ${msg.uci} — engine state has diverged from chess.js`);
      }
      break;
    }
    case "setSims": {
      chessGame.set_simulations(msg.sims);
      break;
    }
    case "think": {
      const { genId, engine, fen, depth, mambaSims, seed } = msg;
      if (engine === "muzero") {
        const result = await chessGame.think(seed);
        postMessage({ type: "thought", genId, uci: result.uci || null, value: result.value });
      } else if (engine === "bee-mamba" && mambaSims > 0) {
        // Value-guided PUCT search (chess_mamba_mcts) on top of the same
        // policy/value heads, instead of just argmaxing the policy head.
        // Sequential on wasm regardless of thread count (see that module's
        // docstring), so 1 thread is passed.
        const t0 = performance.now();
        const uci = (await chessMambaBot.best_move_mcts(fen, mambaSims, 1)) ?? null;
        const dtS = (performance.now() - t0) / 1000;
        console.log(`[bee-mamba mcts] ${mambaSims} sims in ${dtS.toFixed(3)}s -> ${(mambaSims / dtS).toFixed(1)} nodes/s`);
        postMessage({ type: "thought", genId, uci, moves: null, value: null });
      } else if (engine === "bee-mamba") {
        // One forward pass, no search, no history (ChessMamba was trained
        // with no history planes) -- just the current FEN in. Sent as a full
        // ranked list (not just the top move) so the main thread can fall
        // back to the next-best candidate if the top one ever turns out
        // illegal per chess.js's own state (see main.js's `botTurn`).
        const moves = await chessMambaBot.ranked_moves(fen);
        postMessage({ type: "thought", genId, uci: moves[0] ?? null, moves, value: null });
      } else {
        // Synchronous alpha-beta: this call blocks the worker thread for its
        // full duration, but never the main thread — the page stays responsive.
        const uci = chess_bot_move(fen, depth, seed) ?? null;
        postMessage({ type: "thought", genId, uci, value: null });
      }
      break;
    }
  }
};

async function boot() {
  await init();
  chessGame = await create_chess(128);
  chessMambaBot = await create_chess_mamba();
  postMessage({ type: "boot" });
}

boot();
