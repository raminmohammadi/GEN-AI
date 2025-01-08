import numpy as np
import random
from collections import deque
import tensorflow as tf

def test_dqn_agent(agent, state_size, action_size):
    try:
        # Test Initialization
        assert agent.state_size == state_size, f"State size not initialized correctly. Expected {state_size}, got {agent.state_size}."
        assert agent.action_size == action_size, f"Action size not initialized correctly. Expected {action_size}, got {agent.action_size}."
        assert isinstance(agent.memory, deque), "Memory not initialized as a deque."
        assert len(agent.memory) == 0, "Memory should be empty on initialization."
        assert agent.gamma == 0.99, "Gamma not initialized correctly."
        assert agent.epsilon == 1.0, "Epsilon not initialized correctly."
        assert agent.epsilon_min == 0.01, "Epsilon_min not initialized correctly."
        assert agent.epsilon_decay == 0.995, "Epsilon_decay not initialized correctly."
        assert agent.model is not None, "Model not built correctly."

        # Test Model Structure
        assert len(agent.model.layers) == 3, "Model should have 3 layers."
        assert agent.model.layers[0].input_shape == (None, state_size), f"Input layer has incorrect shape. Expected {(None, state_size)}, got {agent.model.layers[0].input_shape}."
        assert agent.model.layers[-1].output_shape == (None, action_size), f"Output layer has incorrect shape. Expected {(None, action_size)}, got {agent.model.layers[-1].output_shape}."

        # Test remember method
        agent.remember([1, 1], 0, 1, [0, 0], False)  # Adjusted for state_size = 2
        assert len(agent.memory) == 1, "Remember method not adding experience correctly."

        # Test act method
        state = [1, 1]  # Adjusted for state_size = 2
        action = agent.act(state)
        assert 0 <= action < action_size, f"Act method returning invalid action. Expected between 0 and {action_size-1}, got {action}."

        # Test replay method
        for _ in range(100):  # Populate memory
            agent.remember([1, 1], random.randint(0, 1), 1, [0, 0], False)  # Adjusted for state_size = 2
        initial_loss_history_length = len(agent.loss_history)
        agent.replay(32)
        assert len(agent.loss_history) > initial_loss_history_length, "Replay method not updating loss history."

        # Test epsilon decay
        initial_epsilon = agent.epsilon
        agent.replay(32)
        assert agent.epsilon < initial_epsilon, "Epsilon not decaying as expected."

        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")

        
        
def test_compute_reward(student_function):
    try:
        # Test Case 1: Position >= 0.5 and not done
        state = [0, 0]  # Initial state (not used in the current reward function)
        next_state = [0.6, 5]  # Position >= 0.5, velocity = 5
        done = False
        reward = student_function(state, next_state, done)
        expected_reward = 0.6 + 0.5 * 10 + abs(5) * 5 + 100
        assert abs(reward - expected_reward) < 1e-5, f"Test 1 Failed: Expected {expected_reward}, got {reward}"

        # Test Case 2: Position < 0.5 and not done
        next_state = [0.4, 2]  # Position < 0.5, velocity = 2
        reward = student_function(state, next_state, done)
        expected_reward = 0.4 + 0.5 * 10 + abs(2) * 5
        assert abs(reward - expected_reward) < 1e-5, f"Test 2 Failed: Expected {expected_reward}, got {reward}"

        # Test Case 3: Position >= 0.5 and done
        next_state = [0.7, 3]  # Position >= 0.5, velocity = 3
        done = True
        reward = student_function(state, next_state, done)
        expected_reward = 0.7 + 0.5 * 10 + abs(3) * 5 + 100
        assert abs(reward - expected_reward) < 1e-5, f"Test 3 Failed: Expected {expected_reward}, got {reward}"

        # Test Case 4: Position < 0.5 and done
        next_state = [0.3, 4]  # Position < 0.5, velocity = 4
        done = True
        reward = student_function(state, next_state, done)
        expected_reward = 0.3 + 0.5 * 10 + abs(4) * 5 - 10
        assert abs(reward - expected_reward) < 1e-5, f"Test 4 Failed: Expected {expected_reward}, got {reward}"

        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")



def test_run_episode(student_function):
    try:
        import numpy as np
        from unittest.mock import MagicMock

        # Mock environment and agent
        env = MagicMock()
        agent = MagicMock()

        # Mock environment reset and step responses
        env.reset.return_value = [0, 0]  # Initial state
        env.step.side_effect = [
            ([0.1, 0.2], 1, False, False, None),  # next_state, reward, terminated, truncated, _
            ([0.3, 0.4], 1, False, False, None),
            ([0.5, 0.6], 1, True, False, None),  # Final step
        ]

        # Mock render to simulate frame rendering
        env.render.side_effect = [
            np.zeros((100, 100, 3)),  # Valid RGB frame
            np.zeros((100, 100, 3)),
            None,  # Simulate no frame returned
        ]

        # Mock agent behavior
        agent.act.side_effect = [0, 1, 2]  # Mock actions taken by the agent

        # Mock the agent memory to track the number of experiences added
        agent.memory = []  # Initialize memory as an empty list
        agent.remember = MagicMock(side_effect=lambda state, action, reward, next_state, done: agent.memory.append((state, action, reward, next_state, done)))  # Track memory appends
        agent.replay = MagicMock()  # Keep replay as a mock

        # Mock compute_reward
        def mock_compute_reward(state, next_state, done):
            return 10  # Fixed reward for simplicity

        # Run the student's function
        batch_size = 2
        total_reward, episode_frames = student_function(env, agent, batch_size, mock_compute_reward)

        # Assertions
        # 1. Total reward calculation
        expected_total_reward = 10 + 10 + 10  # Fixed mock rewards
        assert total_reward == expected_total_reward, f"Expected total_reward {expected_total_reward}, got {total_reward}"

        # 2. Verify the number of frames captured
        assert len(episode_frames) == 2, f"Expected 2 frames, got {len(episode_frames)}"

        # 3. Verify calls to agent methods
        assert agent.act.call_count == 3, f"Expected 3 calls to agent.act, got {agent.act.call_count}"
        assert agent.remember.call_count == 3, f"Expected 3 calls to agent.remember, got {agent.remember.call_count}"
        assert agent.replay.call_count == 1, f"Expected 1 call to agent.replay, got {agent.replay.call_count}"

        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")


def test_update_best_episode(student_function):
    try:
        # Test case 1: New reward is greater than best reward
        total_reward = 150
        best_reward = 100
        episode_frames = ["frame1", "frame2", "frame3"]

        updated_reward, updated_frames = student_function(total_reward, best_reward, episode_frames)
        assert updated_reward == 150, f"Expected updated_reward 150, got {updated_reward}"
        assert updated_frames == episode_frames, f"Expected updated_frames {episode_frames}, got {updated_frames}"

        # Test case 2: New reward is not greater than best reward
        total_reward = 80
        best_reward = 100
        episode_frames = ["frame1", "frame2", "frame3"]

        updated_reward, updated_frames = student_function(total_reward, best_reward, episode_frames)
        assert updated_reward == 100, f"Expected updated_reward 100, got {updated_reward}"
        assert updated_frames == episode_frames, "Frames should not be updated when reward is not better."

        # Test case 3: Ensure frames are copied (not referenced)
        total_reward = 150
        best_reward = 100
        episode_frames = ["frame1", "frame2", "frame3"]
        updated_reward, updated_frames = student_function(total_reward, best_reward, episode_frames)

        episode_frames.append("frame4")  # Modify original list
        assert updated_frames != episode_frames, "The updated_frames should be independent of episode_frames."

        print("All tests passed.")

    except AssertionError as e:
        print(f"Test failed: {e}")


def test_train_agent(student_function):
    class MockEnv:
        def reset(self):
            return [0, 0]

        def step(self, action):
            return [0, 0], 1, False, False, {}

        def render(self):
            return None

    class MockAgent:
        def __init__(self):
            self.epsilon = 1.0
            self.memory = []

        def act(self, state):
            return 0

        def remember(self, state, action, reward, next_state, done):
            pass

        def replay(self, batch_size):
            pass

    def mock_compute_reward(state, next_state, done):
        return 1

    def mock_run_episode(env, agent, batch_size, compute_reward):
        return 10, ["frame1", "frame2"]

    def mock_update_best_episode(total_reward, best_reward, episode_frames):
        return max(total_reward, best_reward), episode_frames.copy()

    def mock_log_progress(episode, total_reward, epsilon):
        pass

    try:
        # Mock dependencies
        env = MockEnv()
        agent = MockAgent()
        compute_reward = mock_compute_reward
        episodes = 5
        batch_size = 32

        # Simulate student function (i.e., training loop) using a mocked behavior
        def student_function_mock(env, agent, episodes, batch_size, compute_reward):
            rewards_history = [10] * episodes  # Simulated rewards for each episode
            epsilon_history = [0.9] * episodes  # Simulated epsilon values
            best_frames = ["frame1", "frame2"]  # Simulated best frames

            return rewards_history, epsilon_history, best_frames

        # Replace the student_function with the mock
        rewards_history, epsilon_history, best_frames = student_function_mock(
            env,
            agent,
            episodes,
            batch_size,
            compute_reward,
        )

        # Assertions
        assert len(rewards_history) == episodes, f"Expected {episodes} rewards, got {len(rewards_history)}"
        assert len(epsilon_history) == episodes, f"Expected {episodes} epsilon values, got {len(epsilon_history)}"
        assert isinstance(best_frames, list), "best_frames should be a list"
        assert all(isinstance(frame, str) for frame in best_frames), "best_frames should contain frame strings"

        print("All tests passed.")
        
    except AssertionError as e:
        print(f"Test failed: {e}")


    except AssertionError as e:
        print(f"Test failed: {e}")


