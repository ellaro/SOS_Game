import tkinter as tk
from tkinter import messagebox, ttk
from main import SOSGame
from puct import PUCTPlayer
from mcts import MCTSPlayer
from network import GameNetwork
import threading


class SOSGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SOS Game - AI vs Human")
        self.root.geometry("700x750")
        self.root.configure(bg='#2c3e50')

        self.game = SOSGame()
        self.selected_letter = 'S'
        self.ai_player = None
        self.game_mode = None
        self.ai_thinking = False  # NEW: track if AI is thinking

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root,
            text="🎮 SOS GAME 🎮",
            font=("Arial", 24, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title.pack(pady=10)

        # Mode selection frame
        self.mode_frame = tk.Frame(self.root, bg='#2c3e50')
        self.mode_frame.pack(pady=5)

        tk.Label(
            self.mode_frame,
            text="Game Mode:",
            font=("Arial", 12, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        ).pack()

        btn_frame = tk.Frame(self.mode_frame, bg='#2c3e50')
        btn_frame.pack(pady=5)

        tk.Button(
            btn_frame,
            text="👤 vs AI (Trained)",
            command=lambda: self.start_game('human_vs_trained'),
            font=("Arial", 10),
            bg='#3498db',
            fg='white',
            width=15,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame,
            text="👤 vs AI (MCTS)",
            command=lambda: self.start_game('human_vs_mcts'),
            font=("Arial", 10),
            bg='#9b59b6',
            fg='white',
            width=15,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame,
            text="👥 vs Human",
            command=lambda: self.start_game('human_vs_human'),
            font=("Arial", 10),
            bg='#2ecc71',
            fg='white',
            width=15,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=3)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Select a game mode to start",
            font=("Arial", 12),
            bg='#34495e',
            fg='#ecf0f1',
            pady=5
        )
        self.status_label.pack(pady=5, fill=tk.X)

        # Letter selection

        self.letter_frame = tk.Frame(self.root, bg='#2c3e50')
        self.letter_frame.pack(pady=5)

        tk.Label(
            self.letter_frame,
            text="Letter:",
            font=("Arial", 11, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        ).pack(side=tk.LEFT, padx=5)

        self.s_button = tk.Button(
            self.letter_frame,
            text="S",
            command=lambda: self.select_letter('S'),
            font=("Arial", 14, "bold"),
            bg='#e74c3c',
            fg='white',
            width=4,
            height=1,
            relief=tk.RAISED,
            bd=2
        )
        self.s_button.pack(side=tk.LEFT, padx=3)

        self.o_button = tk.Button(
            self.letter_frame,
            text="O",
            command=lambda: self.select_letter('O'),
            font=("Arial", 14, "bold"),
            bg='#95a5a6',
            fg='white',
            width=4,
            height=1,
            relief=tk.RAISED,
            bd=2
        )
        self.o_button.pack(side=tk.LEFT, padx=3)

        # Score display
        self.score_frame = tk.Frame(self.root, bg='#2c3e50')
        self.score_frame.pack(pady=5)

        self.score_label = tk.Label(
            self.score_frame,
            text="Player 0: 0  |  Player 1: 0",
            font=("Arial", 13, "bold"),
            bg='#2c3e50',
            fg='#f39c12'
        )
        self.score_label.pack()

        # Board frame
        self.board_frame = tk.Frame(self.root, bg='#34495e', relief=tk.SUNKEN, bd=3)
        self.board_frame.pack(pady=10)

        self.buttons = []
        for r in range(8):
            row = []
            for c in range(8):
                btn = tk.Button(
                    self.board_frame,
                    text="",
                    command=lambda r=r, c=c: self.cell_clicked(r, c),
                    font=("Arial", 16, "bold"),
                    width=3,
                    height=1,
                    bg='#ecf0f1',
                    fg='#2c3e50',
                    relief=tk.RAISED,
                    bd=2
                )
                btn.grid(row=r, column=c, padx=1, pady=1)
                row.append(btn)
            self.buttons.append(row)

        # Control buttons
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(pady=10)

        tk.Button(
            control_frame,
            text="🔄 New Game",
            command=self.reset_game,
            font=("Arial", 11, "bold"),
            bg='#16a085',
            fg='white',
            width=12,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="❌ Exit",
            command=self.root.quit,
            font=("Arial", 11, "bold"),
            bg='#c0392b',
            fg='white',
            width=12,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=5)

        self.select_letter('S')

    def start_game(self, mode):
        self.game_mode = mode
        self.reset_game()

        if mode == 'human_vs_trained':
            try:
                network = GameNetwork.load("network_mcts_20260203_223556.pth")
                # REDUCED simulations from 200 to 100 for faster response
                self.ai_player = PUCTPlayer(network, num_simulations=100, temperature=0)
                self.status_label.config(text="🤖 Playing against Trained AI (Player 1)")
            except:
                messagebox.showerror("Error", "Trained network not found! Using MCTS instead.")
                self.ai_player = MCTSPlayer(num_simulations=50)
                self.status_label.config(text="🤖 Playing against MCTS AI (Player 1)")

        elif mode == 'human_vs_mcts':
            # REDUCED simulations from 100 to 50 for faster response
            self.ai_player = MCTSPlayer(num_simulations=5000)
            self.status_label.config(text="🤖 Playing against MCTS AI (Player 1)")

        elif mode == 'human_vs_human':
            self.ai_player = None
            self.status_label.config(text="👥 Human vs Human - Player 0's turn")

        self.update_board()

    def select_letter(self, letter):
        self.selected_letter = letter
        if letter == 'S':
            self.s_button.config(relief=tk.SUNKEN, bg='#c0392b')
            self.o_button.config(relief=tk.RAISED, bg='#95a5a6')
        else:
            self.o_button.config(relief=tk.SUNKEN, bg='#7f8c8d')
            self.s_button.config(relief=tk.RAISED, bg='#e74c3c')

    def cell_clicked(self, r, c):
        # PREVENT clicks while AI is thinking
        if self.ai_thinking:
            return

        if self.game_mode is None:
            messagebox.showinfo("Info", "Please select a game mode first!")
            return

        if self.game.game_over:
            messagebox.showinfo("Game Over", "Game is over! Start a new game.")
            return

        move = (r, c, self.selected_letter)

        if move not in self.game.legal_moves():
            messagebox.showwarning("Invalid Move", "This cell is already occupied!")
            return

        # Make human move
        self.game.make_move(move)
        self.update_board()

        if self.game.game_over:
            self.show_winner()
            return

        # AI's turn (if not human vs human)
        if self.ai_player and self.game.current_player == 1:
            self.trigger_ai_move()

    def trigger_ai_move(self):
        """Start AI thinking process"""
        self.ai_thinking = True
        self.disable_board()
        self.status_label.config(text="🤖 AI is thinking... ⏳", bg='#e67e22')
        self.root.update()

        # Run AI in thread
        threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        """AI makes a move (runs in separate thread)"""
        try:
            move = self.ai_player.get_move(self.game)
            if move:
                self.game.make_move(move)

            # Update UI in main thread
            self.root.after(0, self.after_ai_move)
        except Exception as e:
            print(f"AI Error: {e}")
            self.root.after(0, self.after_ai_move)

    def after_ai_move(self):
        """Called after AI finishes (runs in main thread)"""
        self.ai_thinking = False
        self.enable_board()
        self.update_board()

        if self.game.game_over:
            self.show_winner()
        # Check if AI gets another turn
        elif self.game.current_player == 1 and self.ai_player:
            # AI created SOS, gets another turn!
            self.root.after(500, self.trigger_ai_move)  # Small delay

    def disable_board(self):
        """Disable all board buttons"""
        for row in self.buttons:
            for btn in row:
                if btn['state'] != tk.DISABLED:
                    btn.config(state=tk.DISABLED)

    def enable_board(self):
        """Enable empty cells"""
        for r in range(8):
            for c in range(8):
                if self.game.board[r][c] is None:
                    self.buttons[r][c].config(state=tk.NORMAL)

    def update_board(self):
        for r in range(8):
            for c in range(8):
                cell = self.game.board[r][c]
                btn = self.buttons[r][c]

                if cell == 'S':
                    btn.config(text='S', bg='#e74c3c', fg='white', state=tk.DISABLED)
                elif cell == 'O':
                    btn.config(text='O', bg='#3498db', fg='white', state=tk.DISABLED)
                else:
                    btn.config(text='', bg='#ecf0f1', fg='#2c3e50', state=tk.NORMAL)

        # Update score
        self.score_label.config(
            text=f"Player 0: {self.game.scores[0]}  |  Player 1: {self.game.scores[1]}"
        )

        # Update status
        if not self.game.game_over and not self.ai_thinking:
            player = self.game.current_player
            if self.game_mode == 'human_vs_human':
                self.status_label.config(text=f"👤 Player {player}'s turn", bg='#34495e')
            elif player == 0:
                self.status_label.config(text="👤 Your turn (Player 0)", bg='#34495e')
            else:
                self.status_label.config(text="🤖 AI's turn (Player 1)", bg='#34495e')

    def show_winner(self):
        winner = self.game.status()

        if winner == 0:
            msg = "🎉 Player 0 Wins! 🎉"
            color = '#2ecc71'
        elif winner == 1:
            msg = "🤖 Player 1 Wins! 🤖"
            color = '#e74c3c'
        else:
            msg = "🤝 It's a Draw! 🤝"
            color = '#f39c12'

        self.status_label.config(text=msg, bg=color)

        messagebox.showinfo(
            "Game Over",
            f"{msg}\n\nScores:\nPlayer 0: {self.game.scores[0]}\nPlayer 1: {self.game.scores[1]}"
        )

    def reset_game(self):
        self.game = SOSGame()
        self.ai_thinking = False
        self.update_board()

        if self.game_mode:
            if self.game_mode == 'human_vs_human':
                self.status_label.config(text="👥 Player 0's turn", bg='#34495e')
            else:
                self.status_label.config(text="👤 Your turn (Player 0)", bg='#34495e')
        else:
            self.status_label.config(text="Select a game mode to start", bg='#34495e')


if __name__ == '__main__':
    root = tk.Tk()
    app = SOSGameGUI(root)
    root.mainloop()