// Runs the wasm engine off the main thread so a slow Alpha-Beta search (fully
// synchronous Rust, no yield points) or a long MuZero search never blocks the
// page's UI thread — only this worker's own thread stalls while it computes.
import init, { create_chess, chess_bot_move } from "../pkg/mz_web.js";

let chessGame = null;

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
      const { genId, engine, fen, depth, seed } = msg;
      if (engine === "muzero") {
        const result = await chessGame.think(seed);
        postMessage({ type: "thought", genId, uci: result.uci || null, value: result.value });
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
  postMessage({ type: "boot" });
}

boot();
