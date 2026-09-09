import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/dist/esm/chess.js";

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
let worker = null;
let genId = 0;
let pendingThink = null; // { genId, resolve } for the in-flight "think" request, if any

const FILES = "abcdefgh";
// Lichess's own cburnett SVG piece set, vendored under ./pieces (see
// pieces/LICENSE.txt) -- filenames are `${color}${TYPE}.svg`, e.g. `wP.svg`.
function pieceIconUrl(piece) {
  return `./pieces/${piece.color}${piece.type.toUpperCase()}.svg`;
}

let chess = new Chess();
let humanColor = "w";
let whiteAtBottom = true;
// Starts busy so clicks are inert until `main()` finishes loading the wasm
// bot and runs the first `newGame()`, which clears it.
let busy = true;
let selected = null; // algebraic square with a piece currently selected
let legalTargets = []; // verbose move objects for the selected piece
let pendingPromotion = null; // { from, to } awaiting a piece choice
let lastMove = null; // { from, to } of the most recently committed move, for the board highlight

function humanTurn() {
  return chess.turn() === humanColor;
}

// Maps a DOM cell's (row, col) — row 0 at the top of the board as displayed —
// to the algebraic square shown there, accounting for board orientation.
function squareAt(row, col) {
  const file = whiteAtBottom ? FILES[col] : FILES[7 - col];
  const rank = whiteAtBottom ? 8 - row : row + 1;
  return `${file}${rank}`;
}

function squareColour(square) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return (file + rank) % 2 === 0 ? "dark" : "light";
}

const cells = Array.from({ length: 64 }, (_, i) => {
  const cell = document.createElement("button");
  cell.className = "cell";
  cell.addEventListener("click", () => onCellClick(i));
  boardEl.appendChild(cell);
  return cell;
});

function kingSquare(color) {
  for (const row of chess.board()) {
    for (const sq of row) {
      if (sq && sq.type === "k" && sq.color === color) {
        return sq.square;
      }
    }
  }
  return null;
}

function render() {
  const inCheck = chess.isCheck();
  const checkSquare = inCheck ? kingSquare(chess.turn()) : null;
  const clickable = humanTurn() && !busy && !chess.isGameOver() && !pendingPromotion;
  const legalSquares = new Set(legalTargets.map((m) => m.to));
  const captureSquares = new Set(legalTargets.filter((m) => m.captured).map((m) => m.to));

  cells.forEach((cell, i) => {
    const row = Math.floor(i / 8);
    const col = i % 8;
    const square = squareAt(row, col);
    const piece = chess.get(square);

    cell.className = `cell ${squareColour(square)}`;
    cell.classList.toggle("selected", square === selected);
    cell.classList.toggle("legal", clickable && legalSquares.has(square));
    cell.classList.toggle("capture", clickable && captureSquares.has(square));
    cell.classList.toggle("check", square === checkSquare);
    cell.classList.toggle("last-move", lastMove != null && (square === lastMove.from || square === lastMove.to));

    // Coordinate labels (see style.css): rank digits on the leftmost
    // column, file letters on the bottom row, regardless of orientation.
    if (col === 0) {
      cell.dataset.rank = square[1];
    } else {
      delete cell.dataset.rank;
    }
    if (row === 7) {
      cell.dataset.file = square[0];
    } else {
      delete cell.dataset.file;
    }

    cell.innerHTML = piece
      ? `<span class="piece" style="background-image: url('${pieceIconUrl(piece)}')" aria-label="${piece.color === "w" ? "white" : "black"} ${piece.type}"></span>`
      : "";
  });

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
    statusEl.textContent = inCheck ? "your move — you're in check" : "your move";
  } else {
    statusEl.textContent = "bot to move";
  }
}

function selectSquare(square) {
  const piece = chess.get(square);
  if (!piece || piece.color !== humanColor) {
    return;
  }
  selected = square;
  legalTargets = chess.moves({ square, verbose: true });
  render();
}

function clearSelection() {
  selected = null;
  legalTargets = [];
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

async function attemptMove(from, to) {
  const candidates = legalTargets.filter((m) => m.to === to);
  if (candidates.length === 0) {
    return;
  }
  if (candidates.length > 1) {
    // Multiple candidates for the same from/to only happens on promotion,
    // where chess.js expands one move into one entry per promotion piece.
    pendingPromotion = { from, to };
    clearSelection();
    render();
    return;
  }
  const move = chess.move(candidates[0]);
  mirrorMove(move);
  lastMove = { from: move.from, to: move.to };
  clearSelection();
  evalEl.textContent = "";
  setBotWarning(null);
  render();
  await settle();
}

async function onCellClick(i) {
  if (busy || chess.isGameOver() || pendingPromotion) {
    return;
  }
  const row = Math.floor(i / 8);
  const col = i % 8;
  const square = squareAt(row, col);

  if (!humanTurn()) {
    return;
  }

  if (selected === square) {
    clearSelection();
    render();
    return;
  }

  if (selected && legalTargets.some((m) => m.to === square)) {
    await attemptMove(selected, square);
    return;
  }

  selectSquare(square);
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
// verbose move objects, so it's applied the exact same way a human's
// click-to-move is.
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
  whiteAtBottom = humanColor === "w";
  clearSelection();
  pendingPromotion = null;
  lastMove = null;
  busy = false;
  evalEl.textContent = "";
  setBotWarning(null);
  render();
  await settle();
}

newGameEl.addEventListener("click", () => {
  if (!busy) {
    newGame();
  }
});
colourEl.addEventListener("change", () => newGame());
engineEl.addEventListener("change", () => {
  // Bee-Mamba is a single forward pass -- no depth (alpha-beta) or
  // simulation count (MuZero) to tune, so both stay hidden for it.
  strengthLabelEl.hidden = engineEl.value !== "alphabeta";
  simsLabelEl.hidden = engineEl.value !== "muzero";
});

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
  backendEl.textContent = "alpha-beta search or the trained MuZero agent, both via WASM";
  await newGame();
}

main().catch((error) => {
  backendEl.textContent = `failed to start: ${error}`;
  console.error(error);
});
