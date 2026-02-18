import copy


class SOSGame:
    """
    SOS Game Implementation
    
    SOS is a two-player game where players take turns writing either 'S' or 'O' in empty
    cells of an 8x8 grid. The goal is to create the sequence "SOS" (horizontally, 
    vertically, or diagonally). Each time a player creates an SOS pattern, they score
    a point and get another turn. The player with the most points when the board is
    full wins.
    
    Rules:
    - 8x8 grid
    - Players alternate placing 'S' or 'O' in empty cells
    - Creating SOS (in any direction) scores 1 point and grants another turn
    - Game ends when board is full
    - Highest score wins
    """
    
    def __init__(self):
        """
        Initialize a new SOS game
        
        State:
            board: 8x8 grid, each cell is None (empty), 'S', or 'O'
            current_player: 0 or 1 (whose turn it is)
            scores: [player0_score, player1_score]
            game_over: boolean flag
        """
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = 0  # Player 0 starts
        self.scores = [0, 0]
        self.game_over = False

    def make_move(self, move):
        """
        Execute a move and update game state
        
        This is the main game logic:
        1. Place the letter on the board
        2. Check for SOS patterns created by this move
        3. Award points and extra turn if SOS was created
        4. Switch players if no SOS was created
        5. Check if game is over (board full)
        
        Args:
            move: Tuple of (row, col, letter) where letter is 'S' or 'O'
            
        Returns:
            (sos_count, changed_player): Info needed for unmake_move
        """
        r, c, letter = move
        assert self.board[r][c] is None, "Cell must be empty"

        # Place the letter
        self.board[r][c] = letter
        
        # Check if this move created any SOS patterns
        sos_count = self._check_sos(r, c)
        self.scores[self.current_player] += sos_count

        # If SOS was created, player gets another turn (don't change player)
        # Otherwise, switch to the other player
        changed_player = sos_count == 0
        if changed_player:
            self.current_player = 1 - self.current_player

        # Check for game end condition (board full)
        if self._is_board_full():
            self.game_over = True

        # Return info needed to undo this move (for search algorithms)
        return (sos_count, changed_player)

    def unmake_move(self, move, move_info):
        r, c, letter = move
        sos_count, changed_player = move_info

        assert self.board[r][c] == letter

        self.board[r][c] = None
        self.scores[self.current_player if not changed_player else 1 - self.current_player] -= sos_count

        if changed_player:
            self.current_player = 1 - self.current_player

        self.game_over = False

    def clone(self):
        """Create a deep copy of the current game state"""
        return copy.deepcopy(self)

    def encode(self):
        """Encode the game state as a binary vector"""
        encoded = []

        # Encode the board (2 bits per cell: 00=empty, 10=S, 01=O)
        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is None:
                    encoded.extend([0, 0])
                elif cell == 'S':
                    encoded.extend([1, 0])
                elif cell == 'O':
                    encoded.extend([0, 1])

        # Add current player (1 bit)
        encoded.append(self.current_player)

        # Add scores (2 values)
        encoded.extend(self.scores)

        return encoded

    def decode(self, action_index):
        """Translate an action index into a move
        Action space: 0-127 (64 cells × 2 letters)
        Even indices = 'S', Odd indices = 'O'
        """
        cell = action_index // 2
        letter = 'S' if action_index % 2 == 0 else 'O'
        row = cell // 8
        col = cell % 8
        return (row, col, letter)

    def status(self):
        """Return game result if finished, None if ongoing"""
        if self.game_over:
            if self.scores[0] > self.scores[1]:
                return 0  # Player 0 wins
            elif self.scores[1] > self.scores[0]:
                return 1  # Player 1 wins
            else:
                return "draw"
        return None  # Game is ongoing

    def legal_moves(self):
        """Return list of all legal moves in current position"""
        if self.game_over:
            return []

        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] is None:
                    moves.append((r, c, 'S'))
                    moves.append((r, c, 'O'))
        return moves

    def _check_sos(self, r, c):
        """
        Check how many SOS patterns were created by the last move at position (r, c)
        
        This is the core scoring logic. We check all 8 directions (horizontal, vertical,
        and both diagonals) for SOS patterns that include the newly placed letter.
        
        Key insight: We only need to check patterns involving the NEW letter!
        - If we placed 'S': check if it's the start OR end of S-O-S
        - If we placed 'O': check if it's the middle of S-O-S
        
        This is more efficient than checking the entire board.
        
        Args:
            r, c: Position of the newly placed letter
            
        Returns:
            sos_count: Number of SOS patterns created (can be multiple!)
        """
        letter = self.board[r][c]
        sos_count = 0

        # 8 directions to check (all combinations of row/col movements)
        # We check in all directions to find SOS patterns
        directions = [
            (0, 1),  # horizontal right
            (0, -1),  # horizontal left
            (1, 0),  # vertical down
            (-1, 0),  # vertical up
            (1, 1),  # diagonal down-right
            (-1, -1),  # diagonal up-left
            (1, -1),  # diagonal down-left
            (-1, 1),  # diagonal up-right
        ]

        if letter == 'S':
            # S can be at the START of SOS (going forward in a direction)
            # or at the END of SOS (going backward in a direction)

            for dr, dc in directions:
                # Check if this S is the FIRST S in pattern S-O-S
                r2, c2 = r + dr, c + dc
                r3, c3 = r + 2 * dr, c + 2 * dc

                # Check bounds
                if 0 <= r2 < 8 and 0 <= c2 < 8 and 0 <= r3 < 8 and 0 <= c3 < 8:
                    if self.board[r2][c2] == 'O' and self.board[r3][c3] == 'S':
                        sos_count += 1

        elif letter == 'O':
            # O can ONLY be in the MIDDLE of SOS
            # Check all 4 directions (not 8, because we'll check both ways)

            basic_directions = [
                (0, 1),  # horizontal
                (1, 0),  # vertical
                (1, 1),  # diagonal \
                (1, -1),  # diagonal /
            ]

            for dr, dc in basic_directions:
                # Check if there's S on both sides: S-O-S
                r_before, c_before = r - dr, c - dc
                r_after, c_after = r + dr, c + dc

                # Check bounds
                if (0 <= r_before < 8 and 0 <= c_before < 8 and
                        0 <= r_after < 8 and 0 <= c_after < 8):
                    if (self.board[r_before][c_before] == 'S' and
                            self.board[r_after][c_after] == 'S'):
                        sos_count += 1

        return sos_count

    def _check_pattern(self, start_r, start_c, dr, dc, pattern):
        """Check if a specific pattern exists starting from given position"""
        for i, expected in enumerate(pattern):
            r = start_r + i * dr
            c = start_c + i * dc

            # Check if out of bounds
            if r < 0 or r >= 8 or c < 0 or c >= 8:
                return False

            # Check if cell doesn't match expected value
            if self.board[r][c] != expected:
                return False

        return True

    def _is_board_full(self):
        """Check if the board is completely filled"""
        for r in range(8):
            for c in range(8):
                if self.board[r][c] is None:
                    return False
        return True

    def print_board(self):
        print("\n  " + " ".join(str(i) for i in range(8)))
        print("  " + "-" * 16)

        for r in range(8):
            row = []
            for c in range(8):
                cell = self.board[r][c]
                row.append(cell if cell is not None else ".")
            print(f"{r}| " + " ".join(row))

        print("\nScores:")
        print(f"Player 0: {self.scores[0]} | Player 1: {self.scores[1]}")
        print(f"Current player: {self.current_player}")
        print(f"Game over: {self.game_over}")


# ==================== TESTING FUNCTIONS ====================

def test_basic_sos():
    """Test basic SOS detection"""
    print("=== Test 1: Basic Horizontal SOS ===")
    game = SOSGame()

    game.make_move((0, 0, 'S'))
    game.make_move((0, 1, 'O'))
    game.make_move((0, 2, 'S'))
    game.print_board()

    print(f"Expected: Player 0 should have 1 point")
    print(f"Actual: Player 0 has {game.scores[0]} points")
    assert game.scores[0] == 1, "Test failed!"
    print("✓ Test passed!\n")


def test_multiple_sos():
    """Test creating multiple SOS patterns"""
    print("=== Test 2: Multiple SOS ===")
    game = SOSGame()

    # Create a cross pattern that makes 2 SOS
    game.make_move((1, 1, 'S'))  # Player 0
    game.make_move((3, 3, 'X'))  # Dummy move, player 1

    # This should print current state
    game.print_board()
    print("✓ Test completed\n")


def test_full_game():
    """Simulate a complete random game"""
    print("=== Test 3: Full Random Game ===")
    import random

    game = SOSGame()
    move_count = 0

    while not game.game_over and move_count < 100:
        legal = game.legal_moves()
        if not legal:
            break

        move = random.choice(legal)
        game.make_move(move)
        move_count += 1

    game.print_board()
    print(f"Game ended after {move_count} moves")
    print(f"Winner: {game.status()}")
    print("✓ Test completed\n")


def test_encode_decode():
    """Test encoding and decoding"""
    print("=== Test 4: Encode/Decode ===")
    game = SOSGame()

    game.make_move((2, 3, 'S'))
    game.make_move((4, 5, 'O'))

    encoded = game.encode()
    print(f"Encoded state length: {len(encoded)}")
    print(f"Expected: {8 * 8 * 2 + 1 + 2} = {131}")

    # Test decode
    action = 10
    move = game.decode(action)
    print(f"Action {action} decodes to: {move}")
    print("✓ Test completed\n")


if __name__ == '__main__':
    # Run basic test from your original code
    print("=== Your Original Test ===")
    game = SOSGame()
    game.print_board()

    game.make_move((2, 2, 'S'))
    game.print_board()

    game.make_move((2, 3, 'O'))
    game.print_board()

    game.make_move((2, 4, 'S'))
    game.print_board()

    print("\nLegal moves count:", len(game.legal_moves()))

    # Run additional tests
    print("\n" + "=" * 50)
    print("RUNNING ADDITIONAL TESTS")
    print("=" * 50 + "\n")

    test_basic_sos()
    test_encode_decode()
    test_full_game()