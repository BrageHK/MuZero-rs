const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const backendEl = document.getElementById("backend");
const evalEl = document.getElementById("eval");
const countBlackEl = document.getElementById("count-black");
const countWhiteEl = document.getElementById("count-white");
const sideBlackEl = document.getElementById("side-black");
const sideWhiteEl = document.getElementById("side-white");
const nameBlackEl = document.getElementById("name-black");
const nameWhiteEl = document.getElementById("name-white");
const newGameEl = document.getElementById("new-game");
const toggleEl = document.getElementById("toggle-fight");

const fighterEl = { black: document.getElementById("fighter-black"), white: document.getElementById("fighter-white") };
const simsFieldEl = { black: document.getElementById("sims-field-black"), white: document.getElementById("sims-field-white") };
const simsEl = { black: document.getElementById("sims-black"), white: document.getElementById("sims-white") };
const depthFieldEl = { black: document.getElementById("depth-field-black"), white: document.getElementById("depth-field-white") };
const depthEl = { black: document.getElementById("depth-black"), white: document.getElementById("depth-white") };

const PASS = 64;
const FILES = "abcdefgh";
const DEPTH_LABELS = { 1: "Easy", 3: "Medium", 5: "Hard", 10: "Very hard"};

// Standard Othello opening: black d5/e4 (squares 28, 35), white d4/e5 (27, 36).
// Painted immediately on load/new-fight so the board never waits on the
// worker's (possibly delayed, if it's mid-search) confirmation to look reset.
const INITIAL_BOARD = (() => {
  const board = new Array(64).fill(0);
  board[28] = 1;
  board[35] = 1;
  board[27] = -1;
  board[36] = -1;
  return board;
})();

// All UI state below is local to the main thread and updated instantly on
// click — it never waits on the worker, which is the whole point of running
// the engine off-thread. The worker's messages only ever bring it into sync
// a little later; a stale message (genId mismatch) is ignored outright.
let genId = 0;
let running = false;
let busy = false;
let gameOver = false;
let prevCells = null;
const config = { black: null, white: null };
const board = { cells: INITIAL_BOARD, counts: [2, 2], lastMove: 255, blackToMove: true };

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

const cells = Array.from({ length: 64 }, (_, square) => {
  const cell = document.createElement("button");
  cell.className = "cell";
  cell.dataset.square = square;
  boardEl.appendChild(cell);
  return cell;
});

function readConfig(side) {
  const value = fighterEl[side].value;
  if (value === "muzero") {
    return { kind: "muzero", sims: Number(simsEl[side].value) };
  }
  if (value === "alphabeta") {
    return { kind: "bot", depth: Number(depthEl[side].value) };
  }
  return { kind: "bot", depth: 0 };
}

function fighterName(cfg) {
  if (!cfg) {
    return "";
  }
  if (cfg.kind === "muzero") {
    return `MuZero · ${cfg.sims}`;
  }
  return cfg.depth === 0 ? "Random" : `Alpha-Beta · ${DEPTH_LABELS[cfg.depth]} (${cfg.depth})`;
}

function syncFighterUi(side) {
  const value = fighterEl[side].value;
  simsFieldEl[side].hidden = value !== "muzero";
  depthFieldEl[side].hidden = value !== "alphabeta";
}

function label(action) {
  return action === PASS ? "pass" : `${FILES[action % 8]}${Math.floor(action / 8) + 1}`;
}

function updateToggle() {
  toggleEl.textContent = running ? "Stop" : "Start fight";
  toggleEl.classList.toggle("running", running);
  toggleEl.disabled = gameOver;
}

function render() {
  cells.forEach((cell, square) => {
    const owner = board.cells[square];
    const prevOwner = prevCells ? prevCells[square] : undefined;

    cell.classList.toggle("last", square === board.lastMove);

    if (owner === 0) {
      if (cell.firstElementChild) {
        cell.innerHTML = "";
      }
      return;
    }
    if (prevCells && prevOwner === owner) {
      // Unchanged stone: leave its DOM node alone so it doesn't replay its
      // placement animation on every render.
      return;
    }

    const flipped = !!prevCells && prevOwner !== 0 && prevOwner !== owner && square !== board.lastMove;
    let stoneClass = `stone ${owner === 1 ? "black" : "white"}`;
    if (flipped) {
      stoneClass += owner === 1 ? " flip-to-black" : " flip-to-white";
    }
    cell.innerHTML = `<span class="${stoneClass}"></span>`;
    if (flipped) {
      cell.firstElementChild.style.animationDelay = `${chebyshev(square, board.lastMove) * FLIP_STEP_MS}ms`;
    }
  });
  prevCells = Array.from(board.cells);

  const [black, white] = board.counts;
  countBlackEl.textContent = black;
  countWhiteEl.textContent = white;
  nameBlackEl.textContent = fighterName(config.black);
  nameWhiteEl.textContent = fighterName(config.white);
  sideBlackEl.classList.toggle("active", board.blackToMove && !gameOver);
  sideWhiteEl.classList.toggle("active", !board.blackToMove && !gameOver);
  boardEl.parentElement.classList.toggle("busy", busy);
  updateToggle();

  if (gameOver) {
    const blackName = fighterName(config.black);
    const whiteName = fighterName(config.white);
    statusEl.textContent =
      black === white
        ? `draw ${black}–${white}`
        : black > white
          ? `${blackName} wins ${black}–${white}`
          : `${whiteName} wins ${white}–${black}`;
  } else if (busy) {
    const mover = board.blackToMove ? config.black : config.white;
    statusEl.innerHTML = `<span class="spinner"></span> ${fighterName(mover)} thinking…`;
  } else {
    statusEl.textContent = running ? "" : "ready";
  }
}

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "boot") {
    backendEl.textContent = "Gumbel MuZero — SIMD CPU inference via WASM";
    syncFighterUi("black");
    syncFighterUi("white");
    newFight();
    return;
  }

  // Stale reply from a fight the user has already replaced — discard.
  if (msg.genId !== genId) {
    return;
  }

  if (msg.type === "thinking") {
    busy = true;
    render();
    return;
  }

  if (msg.type === "moved") {
    busy = false;
    board.cells = msg.board;
    board.counts = msg.counts;
    board.lastMove = msg.lastMove;
    board.blackToMove = msg.blackToMove;
    gameOver = msg.isOver;
    if (msg.action !== undefined) {
      const mover = msg.side === "black" ? config.black : config.white;
      evalEl.textContent =
        msg.value != null
          ? `${fighterName(mover)} played ${label(msg.action)} · rates its position ${(((1 - msg.value) / 2) * 100).toFixed(0)}%`
          : `${fighterName(mover)} played ${label(msg.action)}`;
    }
    if (gameOver) {
      running = false;
    }
    render();
  }
};

function newFight() {
  genId += 1;
  running = false;
  busy = false;
  gameOver = false;
  config.black = readConfig("black");
  config.white = readConfig("white");
  board.cells = INITIAL_BOARD;
  board.counts = [2, 2];
  board.lastMove = 255;
  board.blackToMove = true;
  prevCells = null;
  evalEl.textContent = "";
  render();
  worker.postMessage({ type: "newFight", genId, black: config.black, white: config.white });
}

newGameEl.addEventListener("click", newFight);

toggleEl.addEventListener("click", () => {
  if (gameOver) {
    return;
  }
  running = !running;
  render();
  worker.postMessage({ type: running ? "start" : "stop", genId });
});

for (const side of ["black", "white"]) {
  fighterEl[side].addEventListener("change", () => {
    syncFighterUi(side);
    newFight();
  });
  simsEl[side].addEventListener("change", () => {
    if (config[side]?.kind === "muzero") {
      config[side].sims = Number(simsEl[side].value);
      worker.postMessage({ type: "updateConfig", genId, side, value: config[side] });
    }
  });
  depthEl[side].addEventListener("change", () => {
    if (config[side]?.kind === "bot") {
      config[side].depth = Number(depthEl[side].value);
      worker.postMessage({ type: "updateConfig", genId, side, value: config[side] });
    }
  });
}

render();
