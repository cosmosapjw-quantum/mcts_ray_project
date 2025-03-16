import numpy as np
import matplotlib.pyplot as plt
from utils.state_utils import TicTacToeState
from utils.state_utils import ConnectFourState
from utils.mcts_utils import apply_gradual_temperature, visualize_temperature_distribution

def test_gradual_temperature_tictactoe():
    """Test and visualize gradual temperature effect on TicTacToe"""
    # Create a sample TicTacToe state
    board = np.zeros(9, dtype=np.int8)
    board[4] = 1  # Center position
    state = TicTacToeState(board)
    
    # Create sample visit counts (center has most visits)
    actions = [0, 1, 2, 3, 5, 6, 7, 8]  # All except center (4)
    visits = np.array([10, 5, 8, 7, 15, 9, 6, 12])
    
    # Test different temperature values
    temperatures = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    # Setup plot
    fig, axes = plt.subplots(1, len(temperatures), figsize=(4*len(temperatures), 4))
    if len(temperatures) == 1:
        axes = [axes]
    
    for i, temp in enumerate(temperatures):
        # Apply gradual temperature
        _, probs = apply_gradual_temperature(state, visits, actions, temp)
        
        # Create a 3x3 grid for visualization
        grid = np.zeros((3, 3))
        for j, action in enumerate(actions):
            grid[action // 3, action % 3] = probs[j]
        # Add the center value (which isn't in actions because it's occupied)
        grid[1, 1] = 0
        
        # Display in the corresponding subplot
        axes[i].imshow(grid, cmap='viridis', vmin=0, vmax=max(probs)*1.1)
        
        # Add text values
        for r in range(3):
            for c in range(3):
                val = grid[r, c]
                if val > 0:
                    axes[i].text(c, r, f'{val:.2f}', 
                               ha='center', va='center', 
                               color='white' if val > 0.3 else 'black')
                    
        # Add visit counts for reference
        for j, action in enumerate(actions):
            r, c = action // 3, action % 3
            axes[i].text(c, r-0.3, f'v:{visits[j]}', 
                       ha='center', va='center', fontsize=8, 
                       color='white' if grid[r, c] > 0.3 else 'black')
        
        axes[i].set_title(f'Temperature: {temp}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('tictactoe_temperature_test.png')
    plt.close()
    
    print("TicTacToe temperature test complete. Results saved to 'tictactoe_temperature_test.png'")

def test_gradual_temperature_connect4():
    """Test and visualize gradual temperature effect on Connect Four"""
    # Create a sample Connect Four state
    state = ConnectFourState()
    
    # Sample visit counts for columns
    actions = [0, 1, 2, 3, 4, 5, 6]  # All columns
    visits = np.array([15, 8, 20, 35, 18, 10, 5])  # Column 3 has most visits
    
    # Test different temperature values
    temperatures = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for temp in temperatures:
        # Apply gradual temperature
        _, probs = apply_gradual_temperature(state, visits, actions, temp)
        
        # Plot the probabilities
        ax.plot(actions, probs, marker='o', label=f'Temp = {temp}')
    
    # Add visit counts as bars in background
    visit_probs = visits / np.sum(visits)
    ax.bar(actions, visit_probs, alpha=0.2, label='Visit Distribution')
    
    # Label the plot
    ax.set_xlabel('Column')
    ax.set_ylabel('Probability')
    ax.set_title('Connect Four: Effect of Temperature on Column Selection')
    ax.set_xticks(actions)
    ax.legend()
    
    plt.savefig('connect4_temperature_test.png')
    plt.close()
    
    print("Connect Four temperature test complete. Results saved to 'connect4_temperature_test.png'")

if __name__ == "__main__":
    test_gradual_temperature_tictactoe()
    test_gradual_temperature_connect4()