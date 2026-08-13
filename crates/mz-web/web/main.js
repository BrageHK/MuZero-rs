import init, { create } from "./pkg/mz_web.js";

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const backendEl = document.getElementById("backend");
const evalEl = document.getElementById("eval");
const countBlackEl = document.getElementById("count-black");
const countWhiteEl = document.getElementById("count-white");
const sideBlackEl = document.getElementById("side-black");
const sideWhiteEl = document.getElementById("side-white");
const undoEl = document.getElementById("undo");
const newGameEl = document.getElementById("new-game");
const colourEl = document.getElementById("colour");
const simsEl = document.getElementById("sims");
const simsValueEl = document.getElementById("sims-value");

const PASS = 64;
const FILES = "abcdefgh";

let game = null;
let humanIsBlack = true;
let busy = false;

const cells = Array.from({ length: 64 }, (_, square) => {
  const cell = document.createElement("button");
  cell.className = "cell";
  cell.dataset.square = square;
  cell.addEventListener("click", () => onCellClick(square));
  boardEl.appendChild(cell);
  return cell;
});

function label(action) {
  return action === PASS ? "pass" : `${FILES[action % 8]}${Math.floor(action / 8) + 1}`;
}

function humanTurn() {
  return game.black_to_move() === humanIsBlack;
}

function render() {
  const board = game.board();
  const legal = game.legal_mask();
  const last = game.last_move();
  const clickable = humanTurn() && !busy && !game.is_over();

  cells.forEach((cell, square) => {
    const owner = board[square];
    cell.className = "cell";
    if (owner !== 0) {
      cell.innerHTML = `<span class="stone ${owner === 1 ? "black" : "white"}"></span>`;
    } else {
      cell.innerHTML = "";
      if (clickable && legal[square]) {
        cell.classList.add("legal");
      }
    }
    if (square === last) {
      cell.classList.add("last");
    }
  });

  const [black, white] = game.counts();
  countBlackEl.textContent = black;
  countWhiteEl.textContent = white;
  sideBlackEl.classList.toggle("active", game.black_to_move() && !game.is_over());
  sideWhiteEl.classList.toggle("active", !game.black_to_move() && !game.is_over());
  undoEl.disabled = busy || !game.can_undo();
  boardEl.parentElement.classList.toggle("busy", busy);

  if (game.is_over()) {
    const humanScore = humanIsBlack ? black : white;
    const agentScore = humanIsBlack ? white : black;
    statusEl.textContent =
      humanScore === agentScore
        ? `draw ${black}–${white}`
        : humanScore > agentScore
          ? `you win ${humanScore}–${agentScore}`
          : `agent wins ${agentScore}–${humanScore}`;
  } else if (busy) {
    statusEl.innerHTML = '<span class="spinner"></span> thinking…';
  } else if (humanTurn()) {
    statusEl.textContent = game.must_pass() ? "no move — passing" : "your move";
  } else {
    statusEl.textContent = "agent to move";
  }
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Search with few simulations can finish in a handful of milliseconds, too
// fast to see. Hold the "thinking" spinner up for a floor so the human's
// stone visibly lands before the agent's does.
const MIN_THINK_MS = 400;

async function agentTurn() {
  busy = true;
  render();
  // Force the human's stone + "thinking" spinner to paint before the search
  // (often synchronous GPU work) blocks the main thread.
  await nextFrame();
  // A fresh seed per move: the Gumbel noise is the agent's exploration.
  const [move] = await Promise.all([
    game.think(1 + Math.floor(Math.random() * 2 ** 40)),
    delay(MIN_THINK_MS),
  ]);
  const { action, value } = move;
  move.free();
  game.play(action);
  const winProb = ((1 - value) / 2) * 100;
  evalEl.textContent = `agent played ${label(action)} · it rates your position ${winProb.toFixed(0)}%`;
  busy = false;
  render();
  await settle();
}

/// Plays out forced passes and hands the turn over until a human move is needed.
async function settle() {
  if (game.is_over() || busy) {
    return;
  }
  if (humanTurn()) {
    if (game.must_pass()) {
      game.play(PASS);
      render();
      await settle();
    }
    return;
  }
  await agentTurn();
}

async function onCellClick(square) {
  if (busy || game.is_over() || !humanTurn() || !game.play(square)) {
    return;
  }
  evalEl.textContent = "";
  render();
  await settle();
}

undoEl.addEventListener("click", async () => {
  if (busy) {
    return;
  }
  game.undo();
  evalEl.textContent = "";
  render();
  await settle();
});

newGameEl.addEventListener("click", async () => {
  if (busy) {
    return;
  }
  game.reset();
  humanIsBlack = colourEl.value === "black";
  evalEl.textContent = "";
  render();
  await settle();
});

colourEl.addEventListener("change", () => newGameEl.click());

simsEl.addEventListener("input", () => {
  simsValueEl.textContent = simsEl.value;
  if (game && !busy) {
    game.set_simulations(Number(simsEl.value));
  }
});

async function main() {
  await init();
  const hasWebGpu = "gpu" in navigator;
  game = await create(Number(simsEl.value));
  simsEl.value = game.simulations();
  simsValueEl.textContent = simsEl.value;
  backendEl.textContent = hasWebGpu
    ? "WebGPU · Gumbel MuZero"
    : "no WebGPU in this browser — SIMD CPU inference via WASM (rebuild with --features flex)";
  render();
  await settle();
}

main().catch((error) => {
  backendEl.textContent = `failed to start: ${error}`;
  console.error(error);
});
