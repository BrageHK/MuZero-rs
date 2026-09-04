use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, File, Piece, Rank, Square};

/// Board cells are bits 0..64, row-major: bit = rank * 8 + file, a1 = bit 0.
pub const BOARD_CELLS: usize = 64;

/// Past positions stacked into the observation, most recent first.
pub const HISTORY_LEN: usize = 8;
/// Per history step: 6 own-piece planes, 6 opponent-piece planes, 2 repetition planes.
pub const PLANES_PER_STEP: usize = 14;
/// Side to move, move count, 4 castling rights, no-progress count.
pub const CONST_PLANES: usize = 7;
pub const TOTAL_PLANES: usize = HISTORY_LEN * PLANES_PER_STEP + CONST_PLANES;
/// 8 history steps of 14 planes, plus 7 constant planes, over an 8x8 board: 119*64.
pub const OBS_LEN: usize = TOTAL_PLANES * BOARD_CELLS;

const QUEEN_DIRS: usize = 8;
const QUEEN_DIST: usize = 7;
const QUEEN_PLANES: usize = QUEEN_DIRS * QUEEN_DIST;
const KNIGHT_PLANES: usize = 8;
const UNDERPROMO_DIRS: usize = 3;
const UNDERPROMO_PIECES: usize = 3;
const UNDERPROMO_PLANES: usize = UNDERPROMO_DIRS * UNDERPROMO_PIECES;
/// AlphaZero's per-square move encoding: 56 queen-like slides + 8 knight jumps + 9 underpromotions.
pub const MOVE_PLANES: usize = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMO_PLANES;
/// 64 origin squares times 73 move types (AlphaZero/MuZero chess action space).
pub const ACTION_SIZE: usize = BOARD_CELLS * MOVE_PLANES;

/// Self-play ply cap. Chess has no natural bound like Othello's fixed cell count;
/// 512 plies comfortably covers the 50-move-rule horizon (100 plies) with margin
/// for games where checks/captures keep resetting the clock.
pub const MAX_STEPS: usize = 512;

/// The 8 queen-move directions, mover-relative (rank, file) deltas. Index 0 is
/// "forward" (toward the opponent's back rank).
const QUEEN_DELTAS: [(i8, i8); QUEEN_DIRS] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

const KNIGHT_DELTAS: [(i8, i8); KNIGHT_PLANES] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];

/// Underpromotions only happen moving forward: straight, or a diagonal capture
/// to either side.
const UNDERPROMO_DELTAS: [(i8, i8); UNDERPROMO_DIRS] = [(1, 0), (1, -1), (1, 1)];
const UNDERPROMO_PIECE_ORDER: [Piece; UNDERPROMO_PIECES] = [Piece::Knight, Piece::Bishop, Piece::Rook];

#[inline(always)]
const fn flip_sq(sq: usize) -> usize {
    sq ^ 56
}

#[inline(always)]
fn sq_index(sq: Square) -> usize {
    sq.to_index()
}

#[inline(always)]
fn index_sq(i: usize) -> Square {
    Square::make_square(Rank::from_index(i / 8), File::from_index(i % 8))
}

/// AlphaZero's per-square move encoding, oriented to the side to move (mirrors
/// `Chess::obs`'s board orientation for black).
pub fn move_to_action(board: &Board, mv: ChessMove) -> usize {
    let flip = board.side_to_move() == Color::Black;
    let src = if flip { flip_sq(sq_index(mv.get_source())) } else { sq_index(mv.get_source()) };
    let dst = if flip { flip_sq(sq_index(mv.get_dest())) } else { sq_index(mv.get_dest()) };
    let dr = (dst / 8) as i8 - (src / 8) as i8;
    let dc = (dst % 8) as i8 - (src % 8) as i8;

    let plane = match mv.get_promotion() {
        Some(promo) if promo != Piece::Queen => {
            let dir3 = UNDERPROMO_DELTAS
                .iter()
                .position(|&d| d == (dr, dc))
                .expect("underpromotion must move one square forward or forward-diagonal");
            let piece_idx = UNDERPROMO_PIECE_ORDER
                .iter()
                .position(|&p| p == promo)
                .expect("underpromotion piece must be knight, bishop or rook");
            QUEEN_PLANES + KNIGHT_PLANES + dir3 * UNDERPROMO_PIECES + piece_idx
        }
        // Queen promotion or no promotion: either a knight jump or a queen-like slide.
        _ => match KNIGHT_DELTAS.iter().position(|&d| d == (dr, dc)) {
            Some(k) => QUEEN_PLANES + k,
            None => queen_plane(dr, dc),
        },
    };
    src * MOVE_PLANES + plane
}

fn queen_plane(dr: i8, dc: i8) -> usize {
    let dist = dr.abs().max(dc.abs());
    let (ndr, ndc) = (dr / dist, dc / dist);
    let dir = QUEEN_DELTAS
        .iter()
        .position(|&d| d == (ndr, ndc))
        .expect("queen-like move must be axis-aligned or diagonal");
    dir * QUEEN_DIST + (dist as usize - 1)
}

/// Inverse of `move_to_action`. Returns `None` when the geometric move would
/// step off the board; the caller still has to check board legality.
pub fn action_to_move(board: &Board, action: usize) -> Option<ChessMove> {
    let src_rel = action / MOVE_PLANES;
    let plane = action % MOVE_PLANES;
    let flip = board.side_to_move() == Color::Black;

    let (dr, dc, promotion) = if plane < QUEEN_PLANES {
        let dir = plane / QUEEN_DIST;
        let dist = (plane % QUEEN_DIST) as i8 + 1;
        let (ddr, ddc) = QUEEN_DELTAS[dir];
        (ddr * dist, ddc * dist, None)
    } else if plane < QUEEN_PLANES + KNIGHT_PLANES {
        let (ddr, ddc) = KNIGHT_DELTAS[plane - QUEEN_PLANES];
        (ddr, ddc, None)
    } else {
        let idx = plane - QUEEN_PLANES - KNIGHT_PLANES;
        let (ddr, ddc) = UNDERPROMO_DELTAS[idx / UNDERPROMO_PIECES];
        (ddr, ddc, Some(UNDERPROMO_PIECE_ORDER[idx % UNDERPROMO_PIECES]))
    };

    let src_rank = (src_rel / 8) as i8;
    let src_file = (src_rel % 8) as i8;
    let dst_rank = src_rank + dr;
    let dst_file = src_file + dc;
    if !(0..8).contains(&dst_rank) || !(0..8).contains(&dst_file) {
        return None;
    }
    let dst_rel = dst_rank as usize * 8 + dst_file as usize;

    let src_abs = if flip { flip_sq(src_rel) } else { src_rel };
    let dst_abs = if flip { flip_sq(dst_rel) } else { dst_rel };
    let src_sq = index_sq(src_abs);
    let dst_sq = index_sq(dst_abs);

    // A plain queen-move plane implies queen promotion when a pawn reaches the
    // mover-relative back rank; underpromotions carry their own piece choice.
    let promotion = promotion.or_else(|| {
        if dst_rank == 7 && board.piece_on(src_sq) == Some(Piece::Pawn) {
            Some(Piece::Queen)
        } else {
            None
        }
    });
    Some(ChessMove::new(src_sq, dst_sq, promotion))
}

/// Absolute (non-mover-relative) piece placement snapshot for one ply, kept for
/// the observation's history planes and for repetition detection.
#[derive(Clone, Copy)]
struct PlyRecord {
    pieces: [[u64; 6]; 2],
    hash: u64,
}

fn snapshot(board: &Board) -> PlyRecord {
    let mut pieces = [[0u64; 6]; 2];
    for (color_idx, &color) in [Color::White, Color::Black].iter().enumerate() {
        for (piece_idx, &piece) in chess::ALL_PIECES.iter().enumerate() {
            pieces[color_idx][piece_idx] = (*board.pieces(piece) & *board.color_combined(color)).0;
        }
    }
    PlyRecord { pieces, hash: board.get_hash() }
}

fn write_piece_planes(obs: &mut [f32], plane_base: usize, piece_bbs: &[u64; 6], flip: bool) {
    for (i, &bb) in piece_bbs.iter().enumerate() {
        let mut bb = bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let idx = if flip { flip_sq(sq) } else { sq };
            obs[plane_base + i * 64 + idx] = 1.0;
        }
    }
}

/// Result of `Chess::apply`, mirroring burn-rl's `StepResult` without depending on it.
pub struct StepOutcome {
    pub state: ChessState,
    pub reward: f64,
    pub done: bool,
}

/// Flat observation, length `OBS_LEN`. Chess has no small mover-relative state
/// like Othello's two bitboards (the observation depends on up to 8 plies of
/// history), so the "state" is just the built observation.
pub type ChessState = Vec<f32>;

/// AlphaZero-style chess: `Board::status` plus a game-length repetition/50-move
/// tracker `chess` doesn't keep for you. Observation and action space are both
/// oriented to the side to move: black's view is rank-mirrored (files
/// untouched), so the network only ever has to learn one perspective.
#[derive(Clone)]
pub struct Chess {
    board: Board,
    /// Absolute piece placement per ply since the start of the game, oldest
    /// first; also doubles as the repetition-count source.
    history: Vec<PlyRecord>,
    /// Resets on a pawn move or a capture (including en passant); 100 triggers
    /// the 50-move-rule draw.
    halfmove_clock: u32,
}

impl Default for Chess {
    fn default() -> Self {
        let board = Board::default();
        Self { history: vec![snapshot(&board)], board, halfmove_clock: 0 }
    }
}

impl Chess {
    pub fn new() -> Self {
        Self::default()
    }

    /// Board from an arbitrary position, with fresh history. Only for tests
    /// and puzzle setups.
    pub fn from_board(board: Board) -> Self {
        Self { history: vec![snapshot(&board)], board, halfmove_clock: 0 }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    /// 1-based, incremented after each black move, as in FEN.
    pub fn fullmove_number(&self) -> u32 {
        ((self.history.len() - 1) / 2) as u32 + 1
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn status(&self) -> BoardStatus {
        self.board.status()
    }

    pub fn is_fifty_move_draw(&self) -> bool {
        self.halfmove_clock >= 100
    }

    pub fn is_threefold_repetition(&self) -> bool {
        let hash = self.board.get_hash();
        self.history.iter().filter(|r| r.hash == hash).count() >= 3
    }

    /// Board-wide "dead position" check for the combinations no legal
    /// sequence of moves can checkmate from: bare kings, a lone knight or
    /// bishop against a bare king, or bishops (one per side) confined to a
    /// single square color. Mirrors chess.js's `isInsufficientMaterial`.
    pub fn is_insufficient_material(&self) -> bool {
        let board = &self.board;
        let heavy = *board.pieces(Piece::Pawn) | *board.pieces(Piece::Rook) | *board.pieces(Piece::Queen);
        if heavy != BitBoard(0) {
            return false;
        }

        let bishops = *board.pieces(Piece::Bishop);
        let minors = board.pieces(Piece::Knight).popcnt() + bishops.popcnt();
        match minors {
            0 | 1 => true,
            2 => bishops.popcnt() == 2 && Self::bishops_same_color(bishops),
            _ => false,
        }
    }

    fn bishops_same_color(bishops: BitBoard) -> bool {
        const LIGHT_SQUARES: BitBoard = BitBoard(0x55AA_55AA_55AA_55AA);
        (bishops & LIGHT_SQUARES) == BitBoard(0) || (bishops & !LIGHT_SQUARES) == BitBoard(0)
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        !matches!(self.status(), BoardStatus::Ongoing)
            || self.is_fifty_move_draw()
            || self.is_threefold_repetition()
            || self.is_insufficient_material()
    }

    pub fn is_legal(&self, action: usize) -> bool {
        action < ACTION_SIZE
            && action_to_move(&self.board, action).is_some_and(|mv| self.board.legal(mv))
    }

    /// Decode `action` into the move it represents in the current position
    /// (regardless of legality); for UIs that need to display or apply an
    /// arbitrary action.
    pub fn action_to_move(&self, action: usize) -> Option<ChessMove> {
        action_to_move(&self.board, action)
    }

    /// Encode a move already known to apply to the current position.
    pub fn move_to_action(&self, mv: ChessMove) -> usize {
        move_to_action(&self.board, mv)
    }

    /// Legal actions in the order `chess::MoveGen` produces them (unordered).
    pub fn legal_actions(&self) -> impl Iterator<Item = usize> + '_ {
        chess::MoveGen::new_legal(&self.board).map(|mv| move_to_action(&self.board, mv))
    }

    pub fn legal_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SIZE];
        for action in self.legal_actions() {
            mask[action] = true;
        }
        mask
    }

    /// 119 planes of 8x8, side-to-move relative: `HISTORY_LEN` steps of own/opp
    /// piece planes plus repetition indicators (most recent first, zero-padded
    /// at game start), then 7 constant planes for side to move, move count,
    /// castling rights and the no-progress count. See Table S1 of the AlphaZero
    /// paper (also Leela Chess Zero's classical input format).
    pub fn obs(&self) -> Vec<f32> {
        let mut obs = vec![0.0f32; OBS_LEN];
        let flip = self.side_to_move() == Color::Black;
        let mover = self.side_to_move().to_index();
        let opp = (!self.side_to_move()).to_index();

        let n = self.history.len();
        for step in 0..n.min(HISTORY_LEN) {
            let idx = n - 1 - step;
            let rec = &self.history[idx];
            let base = step * PLANES_PER_STEP * 64;
            write_piece_planes(&mut obs, base, &rec.pieces[mover], flip);
            write_piece_planes(&mut obs, base + 6 * 64, &rec.pieces[opp], flip);

            let occurrences_before = self.history[..idx].iter().filter(|r| r.hash == rec.hash).count();
            let rep_base = base + 12 * 64;
            if occurrences_before >= 1 {
                obs[rep_base..rep_base + 64].fill(1.0);
            }
            if occurrences_before >= 2 {
                obs[rep_base + 64..rep_base + 128].fill(1.0);
            }
        }

        let const_base = HISTORY_LEN * PLANES_PER_STEP * 64;
        if self.side_to_move() == Color::White {
            obs[const_base..const_base + 64].fill(1.0);
        }
        obs[const_base + 64..const_base + 128].fill(self.fullmove_number() as f32 / 100.0);

        let own_castle = self.board.castle_rights(self.side_to_move());
        let opp_castle = self.board.castle_rights(!self.side_to_move());
        obs[const_base + 128..const_base + 192].fill(own_castle.has_kingside() as u8 as f32);
        obs[const_base + 192..const_base + 256].fill(own_castle.has_queenside() as u8 as f32);
        obs[const_base + 256..const_base + 320].fill(opp_castle.has_kingside() as u8 as f32);
        obs[const_base + 320..const_base + 384].fill(opp_castle.has_queenside() as u8 as f32);
        obs[const_base + 384..const_base + 448].fill(self.halfmove_clock as f32 / 100.0);

        obs
    }

    pub fn state(&self) -> ChessState {
        self.obs()
    }

    /// Play `action` for the side to move. Reward is from the perspective of the
    /// player taking the action, granted only on the terminal step: +1 win, -1
    /// loss, 0 draw. Illegal moves end the game at -1, matching Othello's
    /// `apply` convention.
    pub fn apply(&mut self, action: usize) -> StepOutcome {
        debug_assert!(self.is_legal(action), "illegal action {action}\n{self}");
        let Some(mv) = action_to_move(&self.board, action).filter(|&mv| self.board.legal(mv)) else {
            return StepOutcome { state: self.state(), reward: -1.0, done: true };
        };

        let moving_piece = self.board.piece_on(mv.get_source());
        let is_pawn_move = moving_piece == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(mv.get_dest()).is_some()
            || (is_pawn_move && mv.get_source().get_file() != mv.get_dest().get_file());

        self.board = self.board.make_move_new(mv);
        self.halfmove_clock = if is_pawn_move || is_capture { 0 } else { self.halfmove_clock + 1 };
        self.history.push(snapshot(&self.board));

        let done = self.is_over();
        let reward = if done && self.status() == BoardStatus::Checkmate { 1.0 } else { 0.0 };
        StepOutcome { state: self.state(), reward, done }
    }
}

impl core::fmt::Display for Chess {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn perft(board: &Board, depth: usize) -> usize {
        if depth == 0 {
            return 1;
        }
        let moves: Vec<ChessMove> = chess::MoveGen::new_legal(board).collect();
        if depth == 1 {
            return moves.len();
        }
        moves.iter().map(|&mv| perft(&board.make_move_new(mv), depth - 1)).sum()
    }

    #[test]
    fn initial_position_has_20_legal_moves() {
        let game = Chess::new();
        assert_eq!(game.legal_actions().count(), 20);
        assert_eq!(game.legal_mask().iter().filter(|&&l| l).count(), 20);
    }

    #[test]
    fn perft_matches_textbook_counts() {
        // Starting position, depths 1-3 (standard perft reference values).
        let start = Board::default();
        assert_eq!(perft(&start, 1), 20);
        assert_eq!(perft(&start, 2), 400);
        assert_eq!(perft(&start, 3), 8902);

        // "Kiwipete" position, a standard perft stress test full of castling,
        // promotions and en passant.
        let kiwipete = Board::from_str(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .expect("valid FEN");
        assert_eq!(perft(&kiwipete, 1), 48);
        assert_eq!(perft(&kiwipete, 2), 2039);
    }

    #[test]
    fn round_trip_encodes_every_legal_move_from_several_positions() {
        let positions = [
            Board::default(),
            Board::from_str("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap(),
            // White pawn one step from promoting, with an available capture-promotion.
            Board::from_str("r6r/1P4k1/8/8/8/8/6K1/8 w - - 0 1").unwrap(),
            // En passant is available for white on d6.
            Board::from_str("8/8/8/2Pp4/8/2K3k1/8/8 w - d6 0 1").unwrap(),
        ];
        for board in positions {
            for mv in chess::MoveGen::new_legal(&board) {
                let action = move_to_action(&board, mv);
                assert!(action < ACTION_SIZE);
                let decoded = action_to_move(&board, action).expect("must decode back to a move");
                assert_eq!(decoded, mv, "round trip failed for {mv} in\n{board}");
            }
        }
    }

    #[test]
    fn is_legal_rejects_out_of_range_and_pseudo_moves() {
        let game = Chess::new();
        assert!(!game.is_legal(ACTION_SIZE));
        // e2-e5 is not a legal pawn move (too far).
        let bogus = move_to_action(
            &Board::default(),
            ChessMove::new(Square::E2, Square::E5, None),
        );
        assert!(!game.is_legal(bogus));
    }

    #[test]
    fn obs_has_expected_length_and_side_to_move_plane() {
        let white = Chess::new();
        let obs = white.obs();
        assert_eq!(obs.len(), OBS_LEN);
        let const_base = HISTORY_LEN * PLANES_PER_STEP * 64;
        assert!(obs[const_base..const_base + 64].iter().all(|&v| v == 1.0));

        let mut black = white.clone();
        black.apply(move_to_action(black.board(), chess::MoveGen::new_legal(black.board()).next().unwrap()));
        let obs = black.obs();
        assert!(obs[const_base..const_base + 64].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn obs_reflects_own_pieces_from_movers_perspective() {
        let mut game = Chess::new();
        // White plays e2-e4.
        let e4 = chess::MoveGen::new_legal(game.board())
            .find(|mv| mv.get_source() == Square::E2 && mv.get_dest() == Square::E4)
            .unwrap();
        game.apply(move_to_action(game.board(), e4));

        let obs = game.obs();
        // Most recent step (step=0), own-pawn plane (piece index 0). Black is now
        // to move and its pieces haven't budged, so all 8 pawns are still present,
        // and the e7 pawn mirrors onto the same relative cell white's e2 pawn
        // would have occupied (own pieces start on the mover's near ranks).
        let own_pawn_count = obs[0..64].iter().filter(|&&v| v == 1.0).count();
        assert_eq!(own_pawn_count, 8);
        assert_eq!(obs[12], 1.0); // black's e7 pawn, mirrored to relative e2

        // White's e4 pawn (its most advanced) mirrors onto relative e5 in the
        // opponent-piece block (piece index 0 there starts at plane offset 6).
        assert_eq!(obs[6 * 64 + 36], 1.0);
    }

    #[test]
    fn fifty_move_rule_and_threefold_repetition_draw() {
        // A spare pawn (untouched by the king shuffle below) keeps
        // `is_insufficient_material` from ending the game before the
        // repetition draw is reached.
        let board = Board::from_str("7k/8/8/8/3P4/8/8/K7 w - - 0 1").unwrap();
        let mut game = Chess::from_board(board);

        let find = |game: &Chess, from: Square, to: Square| -> usize {
            let mv = chess::MoveGen::new_legal(game.board())
                .find(|mv| mv.get_source() == from && mv.get_dest() == to)
                .expect("move must be legal");
            move_to_action(game.board(), mv)
        };

        let mut done = false;
        'outer: for _ in 0..10 {
            for (from, to) in [
                (Square::A1, Square::A2),
                (Square::H8, Square::H7),
                (Square::A2, Square::A1),
                (Square::H7, Square::H8),
            ] {
                let action = find(&game, from, to);
                let outcome = game.apply(action);
                if outcome.done {
                    assert_eq!(outcome.reward, 0.0, "repetition must be scored as a draw");
                    done = true;
                    break 'outer;
                }
            }
        }
        assert!(done, "threefold repetition should have ended the game");
        assert!(game.is_threefold_repetition());
    }

    #[test]
    fn checkmate_rewards_the_mover_and_stalemate_is_a_draw() {
        // Fool's mate: fastest possible checkmate.
        let mut game = Chess::new();
        for (from, to) in [
            (Square::F2, Square::F3),
            (Square::E7, Square::E5),
            (Square::G2, Square::G4),
        ] {
            let mv = chess::MoveGen::new_legal(game.board())
                .find(|mv| mv.get_source() == from && mv.get_dest() == to)
                .unwrap();
            let outcome = game.apply(move_to_action(game.board(), mv));
            assert!(!outcome.done);
        }
        let mate = chess::MoveGen::new_legal(game.board())
            .find(|mv| mv.get_source() == Square::D8 && mv.get_dest() == Square::H4)
            .unwrap();
        let outcome = game.apply(move_to_action(game.board(), mate));
        assert!(outcome.done);
        assert_eq!(outcome.reward, 1.0);
        assert_eq!(game.status(), BoardStatus::Checkmate);

        // A textbook stalemate position: black to move, no legal moves, not in check.
        let stalemate_setup =
            Chess::from_board(Board::from_str("k7/8/1Q6/8/8/8/8/7K b - - 0 1").unwrap());
        assert_eq!(stalemate_setup.status(), BoardStatus::Stalemate);
        assert!(stalemate_setup.is_over());
    }

    #[test]
    fn insufficient_material_is_a_draw() {
        let bare_kings = Chess::from_board(Board::from_str("8/8/8/4k3/8/8/8/4K3 w - - 0 1").unwrap());
        assert!(bare_kings.is_insufficient_material());
        assert!(bare_kings.is_over());

        let king_and_bishop_vs_king =
            Chess::from_board(Board::from_str("8/8/8/4k3/8/8/3B4/4K3 w - - 0 1").unwrap());
        assert!(king_and_bishop_vs_king.is_insufficient_material());

        let king_and_knight_vs_king =
            Chess::from_board(Board::from_str("8/8/8/4k3/8/8/3N4/4K3 w - - 0 1").unwrap());
        assert!(king_and_knight_vs_king.is_insufficient_material());

        // Same-colored bishops (one per side, both on light squares) can never deliver mate.
        let same_color_bishops =
            Chess::from_board(Board::from_str("1b2k3/8/8/8/8/8/3B4/4K3 w - - 0 1").unwrap());
        assert!(same_color_bishops.is_insufficient_material());

        // Opposite-colored bishops (one per side) are not a dead position.
        let opposite_color_bishops =
            Chess::from_board(Board::from_str("2b1k3/8/8/8/8/8/3B4/4K3 w - - 0 1").unwrap());
        assert!(!opposite_color_bishops.is_insufficient_material());

        // A lone extra pawn is enough material to keep playing.
        let king_and_pawn_vs_king =
            Chess::from_board(Board::from_str("8/8/8/4k3/8/8/3P4/4K3 w - - 0 1").unwrap());
        assert!(!king_and_pawn_vs_king.is_insufficient_material());

        // Bishop + knight together can force mate, unlike either alone.
        let king_bishop_knight_vs_king =
            Chess::from_board(Board::from_str("8/8/8/4k3/8/2N5/3B4/4K3 w - - 0 1").unwrap());
        assert!(!king_bishop_knight_vs_king.is_insufficient_material());
    }
}
