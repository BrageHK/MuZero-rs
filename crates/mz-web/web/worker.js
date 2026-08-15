// Runs the wasm engine off the main thread so a slow Alpha-Beta search (fully
// synchronous Rust, no yield points) or a long MuZero search never blocks the
// page's UI thread — only this worker's own thread stalls while it computes.
import init, { create } from "./pkg/mz_web.js";

const PASS = 64;

let game = null;
// The genId of the fight currently being played out. Bumped by every
// "newFight" message so a step() left over from a superseded fight (awaiting
// a search nobody wants anymore) notices the mismatch and drops its result.
let activeGenId = 0;
let running = false;
const config = { black: null, white: null };

function snapshot() {
  return {
    board: game.board(),
    counts: game.counts(),
    lastMove: game.last_move(),
    blackToMove: game.black_to_move(),
    isOver: game.is_over(),
  };
}

async function step(id) {
  if (id !== activeGenId || !running || game.is_over()) {
    return;
  }

  if (game.must_pass()) {
    game.play(PASS);
    postMessage({ type: "moved", genId: id, ...snapshot() });
    setTimeout(() => step(id), 0);
    return;
  }

  const side = game.black_to_move() ? "black" : "white";
  const mover = config[side];
  postMessage({ type: "thinking", genId: id, side });

  const seed = 1 + Math.floor(Math.random() * 2 ** 40);
  let action;
  let value = null;

  if (mover.kind === "muzero") {
    game.set_simulations(mover.sims);
    const move = await game.think(seed);
    if (id !== activeGenId) {
      move.free();
      return;
    }
    action = move.action;
    value = move.value;
    move.free();
  } else {
    game.set_opponent(mover.depth);
    // Synchronous alpha-beta: this call blocks the worker thread for its
    // full duration, but never the main thread — the page stays responsive.
    action = game.bot_move(seed);
  }

  if (id !== activeGenId) {
    return;
  }
  game.play(action);
  postMessage({ type: "moved", genId: id, side, action, value, ...snapshot() });

  if (game.is_over()) {
    running = false;
    return;
  }
  setTimeout(() => step(id), 0);
}

onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "newFight": {
      activeGenId = msg.genId;
      running = false;
      config.black = msg.black;
      config.white = msg.white;
      game.reset();
      postMessage({ type: "moved", genId: activeGenId, ...snapshot() });
      break;
    }
    case "start": {
      if (msg.genId !== activeGenId || game.is_over()) {
        return;
      }
      running = true;
      step(activeGenId);
      break;
    }
    case "stop": {
      running = false;
      break;
    }
    case "updateConfig": {
      if (msg.genId === activeGenId && config[msg.side]) {
        config[msg.side] = msg.value;
      }
      break;
    }
  }
};

async function boot() {
  await init();
  game = await create(128);
  postMessage({ type: "boot" });
}

boot();
