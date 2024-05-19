async function makeMove(row, col, player) {
    const response = await fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({row: row, col: col, player: player})
    });
    const data = await response.json();
    if (data.error) {
        alert(data.error);
    } else {
        updateBoard(data.board);
        if (data.winner) {
            alert('Winner: ' + data.winner);
            highlightWinner(data.winningCells);
            location.reload();  // Reload the page to reset the game
        }
    }
}

function updateBoard(board) {
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const cell = document.getElementById('cell-' + i + '-' + j);
            cell.textContent = board[i][j];  // Set the text to 'X' or 'O'
            cell.className = '';  // Reset class name

            // Add class based on the cell content to apply CSS for X and O
            if (board[i][j] === 'X') {
                cell.classList.add('x');
            } else if (board[i][j] === 'O') {
                cell.classList.add('o');
            }
        }
    }
}

function highlightWinner(winningCells) {
    winningCells.forEach(cell => {
        const winningElement = document.getElementById('cell-' + cell[0] + '-' + cell[1]);
        winningElement.classList.add('winner');
    });
}
