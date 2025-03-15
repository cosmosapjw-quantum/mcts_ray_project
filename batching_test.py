# batching_test.py
"""
A simplified test script that focuses on testing batch inference
"""
import ray
import time
import numpy as np
from utils.state_utils import TicTacToeState
from inference.batch_inference_server import BatchInferenceServer

def test_batching():
    """Test the batch inference functionality"""
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    print("Creating BatchInferenceServer...")
    # Create inference server with small cache to force network evaluations
    inference_server = BatchInferenceServer.remote(batch_wait=0.002, cache_size=100)
    
    # Wait for server to initialize
    print("Waiting for server initialization...")
    time.sleep(2)
    
    # Create 32 unique TicTacToe states
    states = []
    base_state = TicTacToeState()
    
    # Generate moves for player 1
    for i in range(9):
        # Skip if cell is already filled
        if base_state.board[i] != 0:
            continue
            
        # Apply move
        state1 = base_state.apply_action(i)
        states.append(state1)
        
        # Generate moves for player 2
        for j in range(9):
            if j == i or state1.board[j] != 0:
                continue
                
            state2 = state1.apply_action(j)
            states.append(state2)
            
            # We only need 32 states
            if len(states) >= 32:
                break
                
        if len(states) >= 32:
            break
    
    # Make sure we have enough states
    if len(states) < 32:
        # Generate more if needed
        while len(states) < 32:
            # Create a random state
            random_state = TicTacToeState()
            moves = np.random.choice(range(9), size=np.random.randint(1, 5), replace=False)
            for move in moves:
                if random_state.board[move] == 0:
                    random_state = random_state.apply_action(move)
            states.append(random_state)
    
    # Truncate to exactly 32 states
    states = states[:32]
    
    print(f"Created {len(states)} unique TicTacToe states")
    
    # Test single inference
    print("\nTesting single inferences...")
    start_time = time.time()
    for state in states[:5]:  # Only test a few
        result = ray.get(inference_server.infer.remote(state))
        policy, value = result
        print(f"Policy shape: {policy.shape}, Value: {value}")
    single_time = time.time() - start_time
    print(f"Single inference time: {single_time:.3f}s")
    
    # Test batch inference
    print("\nTesting batch inference...")
    start_time = time.time()
    results = ray.get(inference_server.batch_infer.remote(states))
    batch_time = time.time() - start_time
    
    # Verify results
    for i, result in enumerate(results[:5]):  # Only print a few
        policy, value = result
        print(f"Batch result {i}: Policy shape: {policy.shape}, Value: {value}")
    
    print(f"Batch inference time: {batch_time:.3f}s")
    print(f"Speedup: {single_time / (batch_time/len(states)):.1f}x (adjusted for batch size)")
    
    # Check server stats
    time.sleep(12)  # Wait for stats to update
    
    # Clean up
    ray.shutdown()
    print("Test completed")

if __name__ == "__main__":
    test_batching()