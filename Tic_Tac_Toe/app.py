from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Initialize the game board
board = [["" for _ in range(3)] for _ in range(3)]


def check_winner():
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "":
            return board[0][i]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "":
        return board[0][2]
    return None


def is_full():
    return all(board[i][j] != "" for i in range(3) for j in range(3))


def minimax(board, depth, is_maximizing):
    winner = check_winner()
    if winner == 'O':  # Assuming AI is 'O'
        return None, 1
    elif winner == 'X':
        return None, -1
    elif is_full():
        return None, 0

    if is_maximizing:
        best_score = float('-inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = 'O'
                    _, score = minimax(board, depth + 1, False)
                    board[i][j] = ""
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move, best_score
    else:
        best_score = float('inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = 'X'
                    _, score = minimax(board, depth + 1, True)
                    board[i][j] = ""
                    if score < best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move, best_score


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    row = data['row']
    col = data['col']
    player = data['player']

    if board[row][col] == "":
        board[row][col] = player
        winner = check_winner()
        if winner:
            reset_board()  # Reset the board for a new game
            return jsonify({'winner': winner, 'board': board})
        if is_full():
            reset_board()  # Reset the board for a new game
            return jsonify({'winner': 'Tie', 'board': board})

        # AI move using minimax
        move, _ = minimax(board, 0, True)
        if move:
            board[move[0]][move[1]] = 'O'
            winner = check_winner()
            if winner:
                reset_board()
                return jsonify({'winner': winner, 'board': board})
            if is_full():
                reset_board()
                return jsonify({'winner': 'Tie', 'board': board})

        return jsonify({'board': board, 'next': 'X'})
    return jsonify({'error': 'Invalid move'}), 400


def reset_board():
    global board
    board = [["" for _ in range(3)] for _ in range(3)]


if __name__ == '__main__':
    app.run(debug=True)
