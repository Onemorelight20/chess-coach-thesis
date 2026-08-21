import chess.pgn


def is_valid_game(game: chess.pgn.Game, min_moves: int = 10) -> bool:
    """
    Checks if a game is valid for analysis (e.g., neither too short nor aborted).
    """
    move_count = sum(1 for _ in game.mainline_moves())
    return move_count >= min_moves
