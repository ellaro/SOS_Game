import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
import queue
from main import SOSGame
from puct import PUCTPlayer
from mcts import MCTSPlayer
from network import GameNetwork
from training import SelfPlayTrainer
from replay_buffer import ReplayBuffer
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        # AlphaZero training state
        self.training_popup = None
        self.training_running = False
        self.training_stop_event = None
        self.training_queue = None
        self.training_thread = None
        self.training_trainer = None
        self.training_buffer = None
        self.training_network = None
        self.training_total_steps = 0
        self.training_completed_steps = 0
        self.training_epochs = 0
        self.training_loss_history = {"policy": [], "value": [], "total": []}
        self.training_progress_var = tk.DoubleVar(value=0.0)

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
            text="👤 vs AI (PUCT)",
            command=lambda: self.start_game('human_vs_puct'),
            font=("Arial", 10),
            bg='#f39c12',
            fg='white',
            width=15,
            height=1,
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=3)

        # Option: whether to load a trained network for PUCT
        self.use_trained_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.mode_frame,
            text="Use trained network for PUCT",
            variable=self.use_trained_var,
            bg='#2c3e50',
            fg='#ecf0f1',
            selectcolor='#2c3e50'
        ).pack(pady=4)

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
            text="🧠 Train AlphaZero",
            command=self.open_alphazero_training_popup,
            font=("Arial", 11, "bold"),
            bg='#8e44ad',
            fg='white',
            width=15,
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
            self.ai_player = MCTSPlayer(num_simulations=50000)
            self.status_label.config(text="🤖 Playing against MCTS AI (Player 1)")

        elif mode == 'human_vs_puct':
            # If user requested, try to load the trained network; otherwise
            # use a fresh random `GameNetwork` so PUCT still runs but doesn't
            # automatically load any file.
            if self.use_trained_var.get():
                try:
                    network = GameNetwork.load("network_mcts_20260203_223556.pth")
                    info_text = "🤖 Playing against PUCT AI (trained network) (Player 1)"
                except Exception:
                    network = GameNetwork()
                    info_text = "🤖 Trained network not found — using random network (Player 1)"
            else:
                network = GameNetwork()
                info_text = "🤖 Playing against PUCT AI (random network) (Player 1)"

            # Use PUCT with moderate sims for GUI responsiveness
            self.ai_player = PUCTPlayer(network, num_simulations=150, temperature=0)
            self.status_label.config(text=info_text)

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
            # log to console only via exception, avoid noisy prints
            import traceback
            traceback.print_exc()
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

    def open_alphazero_training_popup(self):
        if self.training_running:
            messagebox.showinfo("Training", "AlphaZero training is already running.")
            return

        if self.training_popup is not None and self.training_popup.winfo_exists():
            self.training_popup.lift()
            return

        popup = tk.Toplevel(self.root)
        popup.title("Train AlphaZero")
        popup.geometry("900x700")
        popup.configure(bg='#2c3e50')
        popup.transient(self.root)
        popup.grab_set()
        popup.protocol("WM_DELETE_WINDOW", self._close_training_popup)
        self.training_popup = popup

        title = tk.Label(
            popup,
            text="Train AlphaZero",
            font=("Arial", 20, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title.pack(pady=10)

        params_frame = tk.Frame(popup, bg='#2c3e50')
        params_frame.pack(pady=5, fill=tk.X)

        entries = {}
        defaults = [
            ("num_iterations", "3"),
            ("games_per_iteration", "10"),
            ("num_simulations", "100"),
            ("epochs", "5"),
        ]

        for i, (label_text, default_value) in enumerate(defaults):
            row = tk.Frame(params_frame, bg='#2c3e50')
            row.pack(fill=tk.X, padx=20, pady=4)

            tk.Label(
                row,
                text=label_text.replace("_", " ").title() + ":",
                font=("Arial", 11, "bold"),
                bg='#2c3e50',
                fg='#ecf0f1',
                width=20,
                anchor='w'
            ).pack(side=tk.LEFT)

            entry = tk.Entry(row, font=("Arial", 11), width=12)
            entry.insert(0, default_value)
            entry.pack(side=tk.LEFT)
            entries[label_text] = entry

        self.training_status_var = tk.StringVar(value="Ready to train")
        status_label = tk.Label(
            popup,
            textvariable=self.training_status_var,
            font=("Arial", 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        status_label.pack(pady=6)

        progress_frame = tk.Frame(popup, bg='#2c3e50')
        progress_frame.pack(fill=tk.X, padx=20, pady=5)

        self.training_progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            mode='determinate',
            variable=self.training_progress_var,
            maximum=100
        )
        self.training_progress_bar.pack(fill=tk.X)

        self.training_progress_label = tk.Label(
            progress_frame,
            text="0%",
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.training_progress_label.pack(anchor='e')

        chart_frame = tk.Frame(popup, bg='#2c3e50')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.training_figure = Figure(figsize=(7, 4), dpi=100)
        self.training_axis = self.training_figure.add_subplot(111)
        self.training_axis.set_title("AlphaZero Training Loss")
        self.training_axis.set_xlabel("Epoch")
        self.training_axis.set_ylabel("Loss")
        self.training_axis.set_facecolor('#ecf0f1')
        self.training_figure.patch.set_facecolor('#ecf0f1')
        self.training_policy_line, = self.training_axis.plot([], [], label='Policy Loss', color='#3498db')
        self.training_value_line, = self.training_axis.plot([], [], label='Value Loss', color='#e74c3c')
        self.training_axis.legend(loc='best')
        self.training_canvas = FigureCanvasTkAgg(self.training_figure, master=chart_frame)
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.training_canvas.draw()

        button_frame = tk.Frame(popup, bg='#2c3e50')
        button_frame.pack(pady=10)

        start_button = tk.Button(
            button_frame,
            text="Start Training",
            command=lambda: self._start_alphazero_training(entries, start_button, stop_button),
            font=("Arial", 11, "bold"),
            bg='#16a085',
            fg='white',
            width=14,
            relief=tk.RAISED,
            bd=2
        )
        start_button.pack(side=tk.LEFT, padx=6)

        stop_button = tk.Button(
            button_frame,
            text="Stop Training",
            command=self.stop_alphazero_training,
            font=("Arial", 11, "bold"),
            bg='#c0392b',
            fg='white',
            width=14,
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED
        )
        stop_button.pack(side=tk.LEFT, padx=6)

        self.training_entries = entries
        self.training_start_button = start_button
        self.training_stop_button = stop_button

    def _close_training_popup(self):
        if self.training_running:
            self.stop_alphazero_training()
        if self.training_popup is not None and self.training_popup.winfo_exists():
            self.training_popup.destroy()
        self.training_popup = None

    def _start_alphazero_training(self, entries, start_button, stop_button):
        try:
            num_iterations = int(entries["num_iterations"].get())
            games_per_iteration = int(entries["games_per_iteration"].get())
            num_simulations = int(entries["num_simulations"].get())
            epochs = int(entries["epochs"].get())
            if num_iterations <= 0 or games_per_iteration <= 0 or num_simulations <= 0 or epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Please enter positive integers for all training parameters.")
            return

        # Use the current PUCT network if available; otherwise start from a fresh network.
        if isinstance(self.ai_player, PUCTPlayer):
            base_network = self.ai_player.network
        else:
            base_network = GameNetwork()

        self.training_network = base_network
        self.training_trainer = SelfPlayTrainer(network=base_network)
        self.training_buffer = ReplayBuffer(max_size=5000)
        self.training_stop_event = threading.Event()
        self.training_queue = queue.Queue()
        self.training_running = True
        self.training_epochs = epochs
        self.training_total_steps = num_iterations * (epochs + 1)
        self.training_completed_steps = 0
        self.training_loss_history = {"policy": [], "value": [], "total": []}
        self.training_progress_var.set(0)

        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        self.training_status_var.set("Starting AlphaZero training...")

        self.training_thread = threading.Thread(
            target=self._run_alphazero_training,
            args=(num_iterations, games_per_iteration, num_simulations, epochs),
            daemon=True
        )
        self.training_thread.start()
        self.root.after(100, self._poll_training_queue)

    def stop_alphazero_training(self):
        if self.training_stop_event is not None:
            self.training_stop_event.set()
            if hasattr(self, 'training_status_var'):
                self.training_status_var.set("Stopping after the current iteration...")

    def _run_alphazero_training(self, num_iterations, games_per_iteration, num_simulations, epochs):
        try:
            for iteration in range(num_iterations):
                # Stop only between iterations, not mid-iteration.
                if self.training_stop_event is not None and self.training_stop_event.is_set():
                    self.training_queue.put(("stopped", f"Training stopped after iteration {iteration}.", None))
                    return

                self.training_queue.put(("status", f"Generating self-play data ({iteration + 1}/{num_iterations})...", None))

                # AlphaZero self-play: temperature=1.0, with Dirichlet noise inside PUCT.
                new_data = self.training_trainer.generate_puct_games(
                    num_games=games_per_iteration,
                    num_simulations=num_simulations,
                    temperature=1.0,
                    verbose=False
                )
                self.training_buffer.add(new_data)

                training_data = list(self.training_buffer.buffer)
                self.training_queue.put((
                    "status",
                    f"Training network on replay buffer ({iteration + 1}/{num_iterations})...",
                    None
                ))

                for epoch in range(epochs):
                    total_loss, policy_loss, value_loss = self.training_trainer.trainer.train_epoch(
                        training_data,
                        batch_size=64
                    )
                    self.training_queue.put((
                        "loss",
                        {
                            "iteration": iteration + 1,
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "total_loss": total_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                        },
                        None
                    ))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f"network_az_{timestamp}_iter{iteration + 1}.pth"
                self.training_network.save(checkpoint_path)

                self.training_queue.put((
                    "iteration_done",
                    f"Saved checkpoint: {checkpoint_path}",
                    None
                ))

            final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = f"network_az_{final_timestamp}.pth"
            self.training_network.save(final_path)
            self.training_queue.put(("done", f"AlphaZero training complete. Saved {final_path}", self.training_network))
        except Exception as exc:
            self.training_queue.put(("error", str(exc), None))

    def _poll_training_queue(self):
        if self.training_queue is None:
            return

        try:
            while True:
                message_type, payload, extra = self.training_queue.get_nowait()

                if message_type == "status":
                    self.training_status_var.set(payload)


                elif message_type == "loss":
                    self.training_loss_history["total"].append(payload["total_loss"])
                    self.training_loss_history["policy"].append(payload["policy_loss"])
                    self.training_loss_history["value"].append(payload["value_loss"])
                    self.training_completed_steps += 1
                    self._update_training_progress()
                    self._update_training_plot()
                    self.training_status_var.set(
                        f"Iteration {payload['iteration']}/{self.training_total_steps // (self.training_epochs + 1)} - "
                        f"Epoch {payload['epoch']}/{payload['epochs']} | "
                        f"Policy: {payload['policy_loss']:.4f} | Value: {payload['value_loss']:.4f}"
                    )

                elif message_type == "iteration_done":
                    self.training_completed_steps += 1
                    self._update_training_progress()
                    self.training_status_var.set(payload)

                elif message_type == "stopped":
                    self._finalize_training(extra=None, message=payload, stopped=True)
                    return

                elif message_type == "done":
                    self._finalize_training(extra=extra, message=payload, stopped=False)
                    return

                elif message_type == "error":
                    self._finalize_training(extra=None, message=f"Training failed: {payload}", stopped=True)
                    messagebox.showerror("Training Error", payload)
                    return
        except queue.Empty:
            pass

        if self.training_running:
            self.root.after(100, self._poll_training_queue)

    def _update_training_progress(self):
        if self.training_total_steps <= 0:
            return
        percent = min(100.0, (self.training_completed_steps / self.training_total_steps) * 100.0)
        self.training_progress_var.set(percent)
        if hasattr(self, 'training_progress_label'):
            self.training_progress_label.config(text=f"{percent:.0f}%")

    def _update_training_plot(self):
        epochs = list(range(1, len(self.training_loss_history["policy"]) + 1))
        self.training_policy_line.set_data(epochs, self.training_loss_history["policy"])
        self.training_value_line.set_data(epochs, self.training_loss_history["value"])

        if epochs:
            self.training_axis.set_xlim(1, max(2, len(epochs)))
            all_losses = self.training_loss_history["policy"] + self.training_loss_history["value"]
            ymin = min(all_losses)
            ymax = max(all_losses)
            if ymin == ymax:
                ymin -= 0.1
                ymax += 0.1
            self.training_axis.set_ylim(ymin - 0.1, ymax + 0.1)

        self.training_axis.relim()
        self.training_axis.autoscale_view(scalex=False, scaley=False)
        self.training_canvas.draw_idle()

    def _finalize_training(self, extra, message, stopped=False):
        self.training_running = False
        self.training_status_var.set(message)

        if hasattr(self, 'training_start_button'):
            self.training_start_button.config(state=tk.NORMAL)
        if hasattr(self, 'training_stop_button'):
            self.training_stop_button.config(state=tk.DISABLED)

        if extra is not None and isinstance(self.ai_player, PUCTPlayer):
            self.ai_player.network = extra

        if self.training_popup is not None and self.training_popup.winfo_exists():
            self.training_progress_var.set(100 if not stopped else self.training_progress_var.get())

        self.training_queue = None
        self.training_stop_event = None
        self.training_thread = None

        if not stopped:
            self.status_label.config(text="🤖 AlphaZero training complete - PUCT updated", bg='#34495e')
        else:
            self.status_label.config(text="Select a game mode to start", bg='#34495e')


if __name__ == '__main__':
    root = tk.Tk()
    app = SOSGameGUI(root)
    root.mainloop()