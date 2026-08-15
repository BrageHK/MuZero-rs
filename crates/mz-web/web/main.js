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

const PASS = 64;
const FILES = "abcdefgh";

let game = null;
let humanIsBlack = true;
let busy = false;
let prevBoard = null;

const FLIP_STEP_MS = 32;

// Flips ripple outward from the placed stone along the flipped line, so
// stagger each flip by its Chebyshev distance from the move that caused it.
function chebyshev(square, from) {
  if (from == null || from < 0 || from > 63) {
    return 0;
  }
  const ax = square % 8;
  const ay = (square / 8) | 0;
  const bx = from % 8;
  const by = (from / 8) | 0;
  return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
}

function humanTurn() {
  return game.black_to_move() === humanIsBlack;
}

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

function render() {
  const board = game.board();
  const legal = game.legal_mask();
  const last = game.last_move();
  const clickable = humanTurn() && !busy && !game.is_over();

  cells.forEach((cell, square) => {
    const owner = board[square];
    const prevOwner = prevBoard ? prevBoard[square] : undefined;

    cell.classList.toggle("legal", owner === 0 && clickable && !!legal[square]);
    cell.classList.toggle("last", square === last);

    if (owner === 0) {
      if (cell.firstElementChild) {
        cell.innerHTML = "";
      }
      return;
    }
    if (prevBoard && prevOwner === owner) {
      // Unchanged stone: leave its DOM node alone so it doesn't replay its
      // placement animation on every render.
      return;
    }

    const flipped = !!prevBoard && prevOwner !== 0 && prevOwner !== owner && square !== last;
    let stoneClass = `stone ${owner === 1 ? "black" : "white"}`;
    if (flipped) {
      stoneClass += owner === 1 ? " flip-to-black" : " flip-to-white";
    }
    cell.innerHTML = `<span class="${stoneClass}"></span>`;
    if (flipped) {
      cell.firstElementChild.style.animationDelay = `${chebyshev(square, last) * FLIP_STEP_MS}ms`;
    }
  });
  prevBoard = Array.from(board);

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
          : `MuZero wins ${agentScore}–${humanScore}`;
  } else if (busy) {
    statusEl.innerHTML = `<span class="spinner"></span> MuZero thinking…`;
  } else if (humanTurn()) {
    statusEl.textContent = game.must_pass() ? "no move — passing" : "your move";
  } else {
    statusEl.textContent = "MuZero to move";
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
  evalEl.textContent = `MuZero played ${label(action)} · it rates its position ${winProb.toFixed(0)}%`;
  busy = false;
  render();
  await settle();
}

/// Plays out forced passes and hands the turn over until the human needs to move.
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
  prevBoard = null;
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
  prevBoard = null;
  evalEl.textContent = "";
  render();
  await settle();
});

colourEl.addEventListener("change", () => newGameEl.click());

simsEl.addEventListener("change", () => {
  if (game && !busy) {
    game.set_simulations(Number(simsEl.value));
  }
});

async function main() {
  const mod = await import("./pkg/mz_web.js");
  await mod.default();
  game = await mod.create(Number(simsEl.value));
  simsEl.value = String(game.simulations());
  backendEl.textContent = "Gumbel MuZero — SIMD CPU inference via WASM";
  render();
  await settle();
}

main().catch((error) => {
  backendEl.textContent = `failed to start: ${error}`;
  console.error(error);
});
