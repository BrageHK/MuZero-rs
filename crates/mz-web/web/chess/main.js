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

// The board state of record lives here in chess.js. `chess_bot_move` (alpha-beta,
// see `crates/mz-web/src/chess_bot.rs`) is stateless FEN-in/UCI-out and never sees
// more than the current FEN. The MuZero opponent (`crates/mz-web/src/chess_game.rs`)
// is stateful instead — its net was trained on up to 8 plies of history, so every
// committed move (human or bot) is replayed into it via `play_uci` to keep that
// history correct, not just handed the latest FEN.
let wasmBotMove = null;
let chessGame = null;

const FILES = "abcdefgh";
// Both colors use the same (filled) glyph shapes and are told apart purely by
// CSS color — the outline "white" chess codepoints (U+2654-2659) render
// solid/filled in several common Linux fonts (DejaVu, Noto Sans Symbols),
// which made white pieces look black.
const GLYPH = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};

let chess = new Chess();
let humanColor = "w";
let whiteAtBottom = true;
// Starts busy so clicks are inert until `main()` finishes loading the wasm
// bot and runs the first `newGame()`, which clears it.
let busy = true;
let selected = null; // algebraic square with a piece currently selected
let legalTargets = []; // verbose move objects for the selected piece
let pendingPromotion = null; // { from, to } awaiting a piece choice

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

  cells.forEach((cell, i) => {
    const row = Math.floor(i / 8);
    const col = i % 8;
    const square = squareAt(row, col);
    const piece = chess.get(square);

    cell.className = `cell ${squareColour(square)}`;
    cell.classList.toggle("selected", square === selected);
    cell.classList.toggle("legal", clickable && legalSquares.has(square));
    cell.classList.toggle("check", square === checkSquare);

    cell.innerHTML = piece ? `<span class="piece ${piece.color === "w" ? "white" : "black"}">${GLYPH[piece.color][piece.type]}</span>` : "";
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

// Mirrors a move already committed to chess.js into the wasm MuZero engine, so
// its history-dependent observation stays correct regardless of which engine
// is currently selected.
function mirrorMove(move) {
  if (chessGame && !chessGame.play_uci(moveToUci(move))) {
    console.error(`chessGame rejected ${moveToUci(move)} — engine state has diverged from chess.js`);
  }
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
  clearSelection();
  evalEl.textContent = "";
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
    evalEl.textContent = "";
    render();
    await settle();
  });
});

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

let lastValue = null;

async function pickBotMove() {
  const seed = 1 + Math.floor(Math.random() * 2 ** 40);
  if (engineEl.value === "muzero") {
    const result = await chessGame.think(seed);
    lastValue = result.value;
    return uciToMove(result.uci);
  }
  const depth = Number(strengthEl.value);
  lastValue = null;
  return uciToMove(wasmBotMove(chess.fen(), depth, seed));
}

// Search with so few plies finishes instantly — too fast to read as
// "thinking". Hold the busy state up for a floor so the bot's move doesn't
// just flash onto the board.
const MIN_THINK_MS = 400;

async function botTurn() {
  busy = true;
  render();
  const move = await pickBotMove();
  await delay(MIN_THINK_MS);
  busy = false;
  if (move) {
    chess.move(move);
    mirrorMove(move);
    evalEl.textContent = lastValue == null ? `bot played ${move.san}` : `bot played ${move.san} (value ${lastValue.toFixed(2)})`;
  }
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
  chess = new Chess();
  chessGame?.reset();
  humanColor = colourEl.value === "white" ? "w" : "b";
  whiteAtBottom = humanColor === "w";
  clearSelection();
  pendingPromotion = null;
  busy = false;
  evalEl.textContent = "";
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
  const muzero = engineEl.value === "muzero";
  strengthLabelEl.hidden = muzero;
  simsLabelEl.hidden = !muzero;
});

async function main() {
  const mod = await import("../pkg/mz_web.js");
  await mod.default();
  wasmBotMove = mod.chess_bot_move;
  backendEl.textContent = "loading MuZero weights…";
  chessGame = await mod.create_chess(Number(simsEl.value));
  simsEl.addEventListener("change", () => chessGame.set_simulations(Number(simsEl.value)));
  backendEl.textContent = "alpha-beta search or the trained MuZero agent, both via WASM";
  await newGame();
}

main().catch((error) => {
  backendEl.textContent = `failed to start: ${error}`;
  console.error(error);
});
