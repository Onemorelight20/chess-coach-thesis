import chess
import chess.engine
import chess.pgn        # only for the analyze_game annotation, but it must be imported
import math
import os

# Clip before computing centipawn loss: an unclipped forced mate produces
# five-figure "losses" that dominate every average. +/-1000 cp = ten pawns.
EVAL_CAP = 1000
MATE_SCORE = 100000  # sentinel so mate keeps the correct sign before capping

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9}
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


class EngineAnalyzer:
    def __init__(self, engine_path: str = None, time_limit: float = 0.1,
                 depth: int = None, eval_cap: int = EVAL_CAP, multipv: int = 5):
        """
        Stockfish wrapper.

        time_limit -- seconds per position (used when depth is None).
        depth      -- fixed search depth; reproducible across machines.
        eval_cap   -- centipawn clip applied to every evaluation.
        multipv    -- candidate moves ranked per position.
        """
        if engine_path is None:
            engine_path = os.getenv("STOCKFISH_PATH", "stockfish")
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.time_limit = time_limit
        self.depth = depth
        self.eval_cap = eval_cap
        self.multipv = multipv

    def _limit(self):
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.time_limit)

    def _cap(self, cp: int) -> int:
        return max(-self.eval_cap, min(self.eval_cap, cp))

    def _capped_white_cp(self, info) -> int:
        score = info["score"].white()
        cp = score.score(mate_score=MATE_SCORE)
        if cp is None:
            cp = 0
        return max(-self.eval_cap, min(self.eval_cap, cp))

    @staticmethod
    def win_prob(cp: float) -> float:
        """
        Centipawn evaluation (mover's POV) -> win probability in [0, 1], via the
        logistic model used by Lichess and Maia (McIlroy-Young et al., KDD 2020).
        """
        cp = max(-2000.0, min(2000.0, float(cp)))
        return 1.0 / (1.0 + math.exp(-0.00368208 * cp))

    def worse_move_candidates(self, board, mover, multipv: int = 8):
        """
        Legal moves ranked below the engine's best, as
        (move, centipawn_loss, win_prob_loss) triples from `mover`'s perspective.
        """
        legal = board.legal_moves.count()
        if legal == 0:
            return []
        infos = self.engine.analyse(board, self._limit(), multipv=min(multipv, legal))
        if isinstance(infos, dict):  # single-pv safety
            infos = [infos]
        raw, best = [], None
        for d in infos:
            pv = d.get("pv")
            if not pv:
                continue
            sc = d["score"].pov(mover).score(mate_score=MATE_SCORE)
            if sc is None:
                continue
            if best is None:
                best = sc
            raw.append((pv[0], sc))
        if best is None:
            return []
        wp_best = self.win_prob(best)
        rows = []
        for mv, sc in raw:
            loss = best - sc
            if loss > 0:
                rows.append((mv, min(loss, 2 * self.eval_cap), wp_best - self.win_prob(sc)))
        return rows

    @staticmethod
    def _positional_signals(board, mover):
        """Board signals computed for the side that just moved (`mover`)."""
        opp = not mover
        pm = board.piece_map()

        hanging = 0
        pins = 0
        for sq, piece in pm.items():
            if piece.color != mover or piece.piece_type == chess.KING:
                continue
            if board.attackers(opp, sq) and not board.attackers(mover, sq):
                hanging += 1
            if board.is_pinned(mover, sq):
                pins += 1

        ksq = board.king(mover)
        king_danger = 0
        if ksq is not None:
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
                king_danger += len(board.attackers(opp, sq))

        center_control = sum(1 for sq in CENTER_SQUARES if board.attackers(mover, sq))

        wm = sum(PIECE_VALUES.get(p.piece_type, 0) for p in pm.values() if p.color == chess.WHITE)
        bm = sum(PIECE_VALUES.get(p.piece_type, 0) for p in pm.values() if p.color == chess.BLACK)

        return hanging, pins, king_danger, center_control, wm - bm

    def analyze_game(self, game: chess.pgn.Game) -> list:
        """
        Analyze a full game move-by-move with MultiPV; one dict per played move.

        Candidates are ranked in the position *before* each move, giving centipawn
        loss vs the best move, the rank of the move played, and whether the
        position was critical (top-two gap >= 100 cp).
        """
        board = game.board()
        results = []
        ply = 0
        for move_node in game.mainline():
            move = move_node.move
            ply += 1
            mover = board.turn
            is_capture = board.is_capture(move)
            mover_mobility = board.legal_moves.count()  # mover's options before moving

            infos = self.engine.analyse(
                board, self._limit(), multipv=min(self.multipv, max(1, mover_mobility)))
            if isinstance(infos, dict):
                infos = [infos]
            lines = []
            for d in infos:
                pv = d.get("pv")
                sc = d["score"].pov(mover).score(mate_score=MATE_SCORE)
                if pv and sc is not None:
                    lines.append((pv[0], self._cap(sc)))

            if lines:
                best, worst = lines[0][1], lines[-1][1]
                second = lines[1][1] if len(lines) > 1 else best
                rank = next((i + 1 for i, (mv, _) in enumerate(lines) if mv == move), None)
                if rank is None:                       # played move worse than the top-K shown
                    rank = len(lines) + 1
                    ti = self.engine.analyse(board, self._limit(), root_moves=[move])
                    sc = ti["score"].pov(mover).score(mate_score=MATE_SCORE)
                    played = self._cap(sc) if sc is not None else best - 2 * self.eval_cap
                else:
                    played = lines[rank - 1][1]
                cpl = max(0, best - played)
                is_critical = (best - second) >= 100
            else:                                      # no analysis (e.g. stalemate-in-1 edge)
                played, cpl, rank, is_critical = 0, 0, 1, False

            board.push(move)
            hanging, pins, king_danger, center, material = self._positional_signals(board, mover)

            # `played` sits below the clip in the fallback above (best - 2*cap
            # marks an unanalysable move as maximally bad, which is a LOSS and is
            # allowed that range), so clip again where the EVALUATION is stored:
            # eval_cp must honour +/-eval_cap unconditionally, since the feature
            # extractor reads it as a position score. cpl above is unaffected.
            eval_cp = self._cap(played)

            results.append({
                "move": move.uci(),
                "turn": "white" if mover == chess.WHITE else "black",
                "eval_cp": eval_cp if mover == chess.WHITE else -eval_cp,
                "cpl": cpl,
                "move_rank": rank,
                "is_critical": is_critical,
                "is_capture": is_capture,
                "is_check": board.is_check(),
                "piece_count": len(board.piece_map()),
                "mobility": mover_mobility,
                "hanging": hanging,
                "pins": pins,
                "king_danger": king_danger,
                "center_control": center,
                "material": material,
                "clock": move_node.clock(),
                "ply": ply,
            })

        return results

    def close(self):
        self.engine.quit()
