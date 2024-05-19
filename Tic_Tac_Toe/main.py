import tkinter as tk
from tkinter import messagebox


def on_button_click(r, c):
    global human_turn
    if board[r][c]['text'] == " " and human_turn:
        board[r][c]['text'] = "X"
        if check_winner("X"):
            messagebox.showinfo("Game Over", "X wins!")
            reset_board()
        elif is_full():
            messagebox.showinfo("Game Over", "It's a tie!")
            reset_board()
        else:
            human_turn = False
            root.after(500, ai_move)  # Delays AI move to make it perceptible to human


def check_winner(player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if board[i][0]['text'] == board[i][1]['text'] == board[i][2]['text'] == player:
            return True
        if board[0][i]['text'] == board[1][i]['text'] == board[2][i]['text'] == player:
            return True
    if board[0][0]['text'] == board[1][1]['text'] == board[2][2]['text'] == player:
        return True
    if board[0][2]['text'] == board[1][1]['text'] == board[2][0]['text'] == player:
        return True
    return False


def is_full():
    # Check if the board is full
    return all(board[r][c]['text'] != " " for r in range(3) for c in range(3))


def reset_board():
    # Reset the board
    for r in range(3):
        for c in range(3):
            board[r][c]['text'] = " "
    global human_turn
    human_turn = True


def ai_move():
    best_score = -float('inf')
    move = None
    for r in range(3):
        for c in range(3):
            if board[r][c]['text'] == " ":
                board[r][c]['text'] = "O"
                score = minimax(board, False)
                board[r][c]['text'] = " "
                if score > best_score:
                    best_score = score
                    move = (r, c)
    if move:
        board[move[0]][move[1]]['text'] = "O"
        if check_winner("O"):
            messagebox.showinfo("Game Over", "O wins!")
            reset_board()
        elif is_full():
            messagebox.showinfo("Game Over", "It's a tie!")
            reset_board()
        else:
            human_turn = True


def minimax(board, is_maximizing):
    if check_winner("O"):
        return 1
    if check_winner("X"):
        return -1
    if is_full():
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c]['text'] == " ":
                    board[r][c]['text'] = "O"
                    score = minimax(board, False)
                    board[r][c]['text'] = " "
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c]['text'] == " ":
                    board[r][c]['text'] = "X"
                    score = minimax(board, True)
                    board[r][c]['text'] = " "
                    best_score = min(score, best_score)
        return best_score


# Set up the main window
root = tk.Tk()
root.title("Tic Tac Toe")

# Create the board
board = [[None for _ in range(3)] for _ in range(3)]
for r in range(3):
    for c in range(3):
        button = tk.Button(root, text=" ", font=('normal', 40), height=1, width=3,
                           command=lambda r=r, c=c: on_button_click(r, c))
        button.grid(row=r, column=c)
        board[r][c] = button

human_turn = True

root.mainloop()
