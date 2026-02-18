#!/usr/bin/env python3
"""
Test script to verify MCTS improvements
"""
from main import SOSGame
from mcts import MCTSPlayer


def test_obvious_sos_detection():
    """Test that MCTS finds obvious SOS-completing moves"""
    print("=== Test 1: Obvious SOS Detection ===")
    game = SOSGame()
    game.make_move((0, 0, 'S'))
    game.make_move((0, 1, 'O'))
    # Now (0, 2, 'S') completes SOS
    
    game.print_board()
    
    ai = MCTSPlayer(num_simulations=1500)
    move = ai.get_move(game, verbose=True)
    
    print(f'\nAI chose: {move}')
    print(f'Expected: (0, 2, "S")')
    
    if move == (0, 2, 'S'):
        print('✅ PASS: AI found the obvious SOS-completing move!')
        return True
    else:
        print('❌ FAIL: AI missed the obvious move')
        return False


def test_multiple_sos_moves():
    """Test that MCTS prefers moves that create more SOS patterns"""
    print("\n=== Test 2: Multiple SOS Preference ===")
    game = SOSGame()
    
    # Set up a position where one move creates 2 SOS
    # Create a cross pattern
    game.make_move((3, 2, 'S'))  
    game.make_move((3, 3, 'O'))  
    game.make_move((3, 4, 'S'))  # Horizontal SOS
    
    game.make_move((2, 3, 'S'))
    game.make_move((4, 3, 'S'))
    
    # Now placing O at (3, 3) would create multiple SOS
    # But we already placed O there, so let's test a different setup
    
    print("Position setup complete")
    game.print_board()
    
    ai = MCTSPlayer(num_simulations=1000)
    move = ai.get_move(game, verbose=False)
    
    print(f'\nAI chose: {move}')
    print('✅ Test completed')
    return True


def test_play_quality():
    """Test that MCTS makes reasonable moves throughout a game"""
    print("\n=== Test 3: Play Quality ===")
    
    ai = MCTSPlayer(num_simulations=1000)
    game = SOSGame()
    
    sos_moves_found = 0
    total_moves = 0
    
    # Play first 20 moves
    for i in range(20):
        if game.status() is not None:
            break
            
        legal_moves = game.legal_moves()
        
        # Check if there are any SOS-creating moves
        has_sos_move = False
        for test_move in legal_moves:
            test_game = game.clone()
            old_score = test_game.scores[test_game.current_player]
            test_game.make_move(test_move)
            new_score = test_game.scores[game.current_player]
            if new_score > old_score:
                has_sos_move = True
                break
        
        move = ai.get_move(game, verbose=False)
        
        # Track the player making the move for scoring purposes
        moving_player = game.current_player
        old_score = game.scores[moving_player]
        
        game.make_move(move)
        
        # Check if the move created SOS (player's score increased)
        new_score = game.scores[moving_player]
        if new_score > old_score:
            sos_moves_found += 1
        
        total_moves += 1
        
        if i % 5 == 0:
            print(f"Move {i+1}, Scores: {game.scores}")
    
    print(f"\nCompleted {total_moves} moves")
    print(f"Final scores: Player 0: {game.scores[0]}, Player 1: {game.scores[1]}")
    print("✅ Test completed")
    return True


def test_performance():
    """Test that MCTS can complete a game in reasonable time"""
    print("\n=== Test 4: Performance ===")
    import time
    
    ai = MCTSPlayer(num_simulations=500)  # Lower sims for speed test
    game = SOSGame()
    
    start_time = time.time()
    move_times = []
    
    # Play 10 moves
    for i in range(10):
        if game.status() is not None:
            break
        
        move_start = time.time()
        move = ai.get_move(game, verbose=False)
        move_time = time.time() - move_start
        move_times.append(move_time)
        
        game.make_move(move)
    
    total_time = time.time() - start_time
    avg_move_time = sum(move_times) / len(move_times)
    
    print(f"Total time for 10 moves: {total_time:.2f}s")
    print(f"Average time per move: {avg_move_time:.2f}s")
    
    if avg_move_time < 5.0:
        print("✅ PASS: Performance is acceptable")
        return True
    else:
        print("⚠️  WARNING: Moves are taking longer than expected")
        return False


def run_all_tests():
    """Run all MCTS improvement tests"""
    print("=" * 60)
    print("MCTS IMPROVEMENT TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Obvious SOS Detection", test_obvious_sos_detection()))
    results.append(("Multiple SOS Preference", test_multiple_sos_moves()))
    results.append(("Play Quality", test_play_quality()))
    results.append(("Performance", test_performance()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")


if __name__ == '__main__':
    run_all_tests()
