import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/dist/esm/chess.js";
import { Chessground } from "./chessground/chessground.js";

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const backendEl = document.getElementById("backend");
const evalEl = document.getElementById("eval");
const nameWhiteEl = document.getElementById("name-white");
const nameBlackEl = document.getElementById("name-black");
const sideWhiteEl = document.getElementById("side-white");
const sideBlackEl = document.getElementById("side-black");
const newGameEl = document.getElementById("new-game");
const colourEl = document.getElementById("colour");
const engineEl = document.getElementById("engine");
const strengthLabelEl = document.getElementById("strength-label");
const strengthEl = document.getElementById("strength");
const simsLabelEl = document.getElementById("sims-label");
const simsEl = document.getElementById("sims");
const mambaSimsLabelEl = document.getElementById("mamba-sims-label");
const mambaSimsEl = document.getElementById("mamba-sims");
const promotionEl = document.getElementById("promotion");
const botWarningEl = document.getElementById("bot-warning");

// The board state of record lives here in chess.js. Both bots run in
// `worker.js`, off the main thread, so a slow Alpha-Beta search or a long
// MuZero search never blocks the page's UI. `chess_bot_move` (alpha-beta,
// see `crates/mz-web/src/chess_bot.rs`) is stateless FEN-in/UCI-out and never
// sees more than the current FEN. The MuZero opponent
// (`crates/mz-web/src/chess_game.rs`) is stateful instead — its net was
// trained on up to 8 plies of history, so every committed move (human or
// bot) is mirrored into the worker's copy via `play_uci` to keep that
// history correct, not just handed the latest FEN.
//
// The board itself is lichess's own `chessground` (vendored under
// ./chessground/, see its LICENSE.txt) -- it owns all rendering, dragging,
// click-to-move, and highlighting; this file's job is just to keep it in
// sync with chess.js via `syncBoard()`.
let worker = null;
let cg = null;
let genId = 0;
let pendingThink = null; // { genId, resolve } for the in-flight "think" request, if any

let chess = new Chess();
let humanColor = "w";
// Starts busy so clicks are inert until `main()` finishes loading the wasm
// bot and runs the first `newGame()`, which clears it.
let busy = true;
let pendingPromotion = null; // { from, to } awaiting a piece choice
let lastMove = null; // { from, to } of the most recently committed move, for the board highlight

function humanTurn() {
  return chess.turn() === humanColor;
}

function toColor(chessJsColor) {
  return chessJsColor === "w" ? "white" : "black";
}

// Whether the human can move right now -- gates both chessground's
// `movable.dests` (below) and the status text.
function clickable() {
  return humanTurn() && !busy && !chess.isGameOver() && !pendingPromotion;
}

// chessground's `movable.dests` map: every square with a piece that can
// move, to the list of squares it can legally move to. Built fresh from
// chess.js before every `syncBoard()` so it's always exactly chess.js's own
// legal moves -- chessground itself has no chess rules, it just offers
// whatever destinations it's given.
function toDests() {
  const dests = new Map();
  for (const move of chess.moves({ verbose: true })) {
    const list = dests.get(move.from);
    if (list) {
      list.push(move.to);
    } else {
      dests.set(move.from, [move.to]);
    }
  }
  return dests;
}

// Pushes the current chess.js position into chessground: piece placement,
// whose turn it is, check/last-move highlights, and which squares are
// draggable right now. Safe to call any time chess.js is the current
// truth -- NOT while a promotion choice is pending, since chessground has
// already moved the pawn optimistically and re-pushing the (unchanged)
// pre-move fen would revert that move on screen.
function syncBoard() {
  cg.set({
    fen: chess.fen(),
    orientation: toColor(humanColor),
    turnColor: toColor(chess.turn()),
    check: chess.isCheck(),
    lastMove: lastMove ? [lastMove.from, lastMove.to] : undefined,
    movable: {
      color: clickable() ? toColor(humanColor) : undefined,
      dests: clickable() ? toDests() : new Map(),
    },
  });
}

function render() {
  nameWhiteEl.textContent = humanColor === "w" ? "You" : "Bot";
  nameBlackEl.textContent = humanColor === "b" ? "You" : "Bot";
  sideWhiteEl.classList.toggle("active", chess.turn() === "w" && !chess.isGameOver());
  sideBlackEl.classList.toggle("active", chess.turn() === "b" && !chess.isGameOver());
  boardEl.parentElement.classList.toggle("busy", busy);
  promotionEl.hidden = !pendingPromotion;

  if (chess.isCheckmate()) {
    statusEl.textContent = humanTurn() ? "checkmate — bot wins" : "checkmate — you win";
  } else if (chess.isStalemate()) {
    statusEl.textContent = "draw — stalemate";
  } else if (chess.isDraw()) {
    statusEl.textContent = "draw";
  } else if (pendingPromotion) {
    statusEl.textContent = "choose a promotion";
  } else if (busy) {
    statusEl.innerHTML = `<span class="spinner"></span> bot thinking…`;
  } else if (humanTurn()) {
    statusEl.textContent = chess.isCheck() ? "your move — you're in check" : "your move";
  } else {
    statusEl.textContent = "bot to move";
  }
}

// The UCI form of a chess.js verbose move object, e.g. `e2e4`, `e7e8q`.
function moveToUci(move) {
  return move.from + move.to + (move.promotion ?? "");
}

// Mirrors a move already committed to chess.js into the worker's MuZero
// engine, so its history-dependent observation stays correct regardless of
// which engine is currently selected.
function mirrorMove(move) {
  worker.postMessage({ type: "mirrorMove", uci: moveToUci(move) });
}

// chessground's `movable.events.after` callback: fires once a human
// drag/click move lands on a legal destination (`dests`, built from chess.js,
// is the only thing constraining which moves chessground will even offer).
async function onUserMove(orig, dest) {
  const candidates = chess.moves({ verbose: true }).filter((m) => m.from === orig && m.to === dest);
  if (candidates.length > 1) {
    // Multiple candidates for the same from/to only happens on promotion,
    // where chess.js expands one move into one entry per promotion piece.
    // chessground has already moved the pawn on screen; leave chess.js and
    // the board alone (beyond locking further input) until the piece choice
    // comes in below.
    pendingPromotion = { from: orig, to: dest };
    cg.set({ movable: { color: undefined, dests: new Map() } });
    render();
    return;
  }
  const move = chess.move(candidates[0]);
  mirrorMove(move);
  lastMove = { from: move.from, to: move.to };
  evalEl.textContent = "";
  setBotWarning(null);
  syncBoard();
  render();
  await settle();
}

promotionEl.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!pendingPromotion) {
      return;
    }
    const { from, to } = pendingPromotion;
    pendingPromotion = null;
    const move = chess.move({ from, to, promotion: button.dataset.piece });
    mirrorMove(move);
    lastMove = { from: move.from, to: move.to };
    evalEl.textContent = "";
    setBotWarning(null);
    syncBoard();
    render();
    await settle();
  });
});

function setBotWarning(message) {
  botWarningEl.textContent = message ?? "";
  botWarningEl.hidden = message == null;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Parses a `e2e4` / `e7e8q` UCI-style string into one of chess.js's own
// verbose move objects, so it's applied the exact same way a human's move is.
function uciToMove(uci) {
  if (uci == null) {
    return null;
  }
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promotion = uci.length > 4 ? uci[4] : undefined;
  return chess.moves({ square: from, verbose: true }).find((m) => m.to === to && (promotion == null || m.promotion === promotion));
}

// Posts a "think" request to the worker and resolves with its "thought"
// reply. Guarded by genId so a reply for a game superseded by `newGame()`
// while the worker was still computing gets dropped instead of applied.
function requestThink(id) {
  return new Promise((resolve) => {
    pendingThink = { genId: id, resolve };
    const seed = 1 + Math.floor(Math.random() * 2 ** 40);
    worker.postMessage({
      type: "think",
      genId: id,
      engine: engineEl.value,
      fen: chess.fen(),
      depth: Number(strengthEl.value),
      mambaSims: Number(mambaSimsEl.value),
      seed,
    });
  });
}

// Search with so few plies finishes instantly — too fast to read as
// "thinking". Hold the busy state up for a floor so the bot's move doesn't
// just flash onto the board.
const MIN_THINK_MS = 400;

// Picks a legal move out of the bot's reply, falling back to progressively
// later candidates -- and finally a random legal move -- if the top pick(s)
// turn out illegal per chess.js's own legality check. `result.moves` is only
// populated for bee-mamba (a searchless policy-head argmax that can, at the
// margins, disagree with chess.js on what's legal in a position); alpha-beta
// and MuZero keep their original single-uci, no-fallback behavior.
function pickBotMove(result) {
  const candidates = result.moves ?? (result.uci != null ? [result.uci] : []);
  for (const uci of candidates) {
    const move = uciToMove(uci);
    if (move) {
      const warning = uci === candidates[0] ? null : `bee-mamba's top move (${candidates[0]}) was illegal — played its next-best legal move (${uci}) instead.`;
      return { move, warning };
    }
  }
  if (result.moves == null) {
    return { move: null, warning: null };
  }
  const legal = chess.moves({ verbose: true });
  const move = legal[Math.floor(Math.random() * legal.length)];
  return { move, warning: `bee-mamba had no legal move among its candidates — played a random legal move (${move.san}) instead.` };
}

async function botTurn() {
  busy = true;
  render();
  const id = genId;
  const [result] = await Promise.all([requestThink(id), delay(MIN_THINK_MS)]);
  if (id !== genId) {
    return;
  }
  busy = false;
  const { move, warning } = pickBotMove(result);
  if (move) {
    chess.move(move);
    mirrorMove(move);
    lastMove = { from: move.from, to: move.to };
    evalEl.textContent = result.value == null ? `bot played ${move.san}` : `bot played ${move.san} (value ${result.value.toFixed(2)})`;
  }
  setBotWarning(warning);
  syncBoard();
  render();
  await settle();
}

async function settle() {
  if (chess.isGameOver() || busy) {
    return;
  }
  if (!humanTurn()) {
    await botTurn();
  }
}

async function newGame() {
  genId += 1;
  pendingThink = null;
  chess = new Chess();
  worker.postMessage({ type: "reset" });
  humanColor = colourEl.value === "white" ? "w" : "b";
  pendingPromotion = null;
  lastMove = null;
  busy = false;
  evalEl.textContent = "";
  setBotWarning(null);
  syncBoard();
  render();
  await settle();
}

newGameEl.addEventListener("click", () => {
  if (!busy) {
    newGame();
  }
});
colourEl.addEventListener("change", () => newGame());
function syncEngineControls() {
  strengthLabelEl.hidden = engineEl.value !== "alphabeta";
  simsLabelEl.hidden = engineEl.value !== "muzero";
  mambaSimsLabelEl.hidden = engineEl.value !== "bee-mamba";
}
engineEl.addEventListener("change", syncEngineControls);

async function main() {
  backendEl.textContent = "loading MuZero weights…";
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  await new Promise((resolve) => {
    worker.onmessage = (event) => {
      if (event.data.type === "boot") {
        resolve();
      }
    };
  });
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "thought" && pendingThink && msg.genId === pendingThink.genId) {
      const resolve = pendingThink.resolve;
      pendingThink = null;
      resolve(msg);
    }
  };
  worker.postMessage({ type: "setSims", sims: Number(simsEl.value) });
  simsEl.addEventListener("change", () => worker.postMessage({ type: "setSims", sims: Number(simsEl.value) }));
  syncEngineControls();
  backendEl.textContent = "alpha-beta search on CPU, or the MuZero/ChessMamba agents via WebGPU";

  cg = Chessground(boardEl, {
    orientation: toColor(humanColor),
    coordinates: true,
    disableContextMenu: true,
    movable: {
      free: false,
      events: { after: (orig, dest) => onUserMove(orig, dest) },
    },
    premovable: { enabled: false },
    predroppable: { enabled: false },
    highlight: { lastMove: true, check: true },
  });

  await newGame();
}

main().catch((error) => {
  backendEl.textContent = `failed to start: ${error}`;
  console.error(error);
});
