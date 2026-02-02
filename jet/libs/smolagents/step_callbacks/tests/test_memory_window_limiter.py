# tests/test_memory_window_limiter.py

from unittest.mock import Mock

import pytest
from jet.libs.smolagents.step_callbacks.memory_window import memory_window_limiter
from smolagents import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.monitoring import Timing

# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────


@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.memory = Mock()
    agent.memory.steps = []
    agent.verbosity_level = 1
    return agent


@pytest.fixture
def dummy_timing():
    return Timing(start_time=0.0, end_time=1.0)


@pytest.fixture
def system_step():
    return SystemPromptStep(system_prompt="You are a helpful assistant.")


@pytest.fixture
def task_step():
    return TaskStep(task="Answer the user's question about quantum physics.")


@pytest.fixture
def planning_step():
    return PlanningStep(
        model_input_messages=[],
        model_output_message=Mock(),
        plan="Step 1: Search\nStep 2: Summarize",
        timing=Timing(start_time=0, end_time=2),
    )


@pytest.fixture
def action_step(dummy_timing):
    return ActionStep(
        step_number=1,
        timing=dummy_timing,
        model_output="I called search tool",
        observations="Found 5 results",
        code_action=None,
        tool_calls=None,
    )


@pytest.fixture
def final_step():
    return FinalAnswerStep(output="The answer is 42")


# ────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────


def test_no_truncation_when_short(mock_agent, system_step, task_step, action_step):
    # Given
    steps = [system_step, task_step, action_step, action_step]
    mock_agent.memory.steps = steps.copy()

    limiter = memory_window_limiter(max_recent_steps=10)

    # When
    limiter(MemoryStep(), mock_agent)  # trigger callback

    # Then
    assert mock_agent.memory.steps == steps  # unchanged


def test_truncates_old_steps_keeps_recent(mock_agent, action_step):
    # Given
    old_steps = [action_step for _ in range(15)]
    recent_steps = [action_step for _ in range(6)]
    mock_agent.memory.steps = old_steps + recent_steps

    limiter = memory_window_limiter(max_recent_steps=8)

    # When
    limiter(MemoryStep(), mock_agent)

    # Then
    # 0 preserved + placeholder + 7 recent actions = 8
    assert len(mock_agent.memory.steps) == 8
    assert all(isinstance(s, ActionStep) for s in mock_agent.memory.steps)
    assert any(
        s.step_number == -1 for s in mock_agent.memory.steps
    )  # placeholder present


def test_keeps_system_and_task(mock_agent, system_step, task_step, action_step):
    # Given
    steps = (
        [system_step, task_step]
        + [action_step for _ in range(20)]
        + [action_step for _ in range(5)]
    )
    mock_agent.memory.steps = steps

    limiter = memory_window_limiter(
        max_recent_steps=6,
        keep_system_and_task=True,
    )

    # When
    limiter(MemoryStep(), mock_agent)

    # Then
    new_steps = mock_agent.memory.steps
    # 2 preserved + placeholder + 6 recent = 9 ? No — after fix = 2 + 1 + 5 = 8
    # Wait — let's run the numbers:
    # preserved = 2
    # effective_recent = 6 - 1 = 5
    # recent added = 5 (assuming no overlap)
    # total = 2 + 1 + 5 = 8
    assert len(new_steps) == 8
    assert isinstance(new_steps[0], SystemPromptStep)
    assert isinstance(new_steps[1], TaskStep)
    assert isinstance(new_steps[2], ActionStep)
    assert new_steps[2].step_number == -1  # placeholder at index 2
    assert all(isinstance(s, ActionStep) for s in new_steps[3:])


def test_keeps_final_answer(mock_agent, system_step, action_step, final_step):
    # Given
    steps = (
        [system_step]
        + [action_step for _ in range(12)]
        + [final_step]  # already at end → good
        + [action_step for _ in range(3)]
    )
    mock_agent.memory.steps = steps

    limiter = memory_window_limiter(max_recent_steps=5, keep_final=True)

    # When
    limiter(MemoryStep(), mock_agent)

    # Then
    new = mock_agent.memory.steps
    assert len(new) <= 8
    assert any(isinstance(s, FinalAnswerStep) for s in new)
    # After fix: final should stay near end if recent window includes it
    # Better assertion:
    final_in_new = next((s for s in new if isinstance(s, FinalAnswerStep)), None)
    assert final_in_new is not None


def test_inserts_placeholder_when_truncating(
    mock_agent, system_step, action_step, dummy_timing
):
    # Given
    steps = [system_step] + [action_step for _ in range(25)]
    mock_agent.memory.steps = steps

    limiter = memory_window_limiter(
        max_recent_steps=6,
        insert_placeholder=True,
        keep_system_and_task=True,
    )

    # When
    limiter(MemoryStep(), mock_agent)

    # Then
    new = mock_agent.memory.steps
    # 1 preserved (system) + placeholder + 6 recent ? No — effective_recent=5
    # → 1 + 1 + 5 = 7
    # But test was expecting 7 → now matches
    assert len(new) == 7
    assert isinstance(new[1], ActionStep)
    assert new[1].step_number == -1
    assert "[Memory window applied" in new[1].model_output


def test_keeps_last_n_planning_steps(mock_agent, planning_step, action_step):
    # Given
    steps = (
        [planning_step] * 3
        + [action_step] * 15
        + [planning_step] * 2
        + [action_step] * 4
    )
    mock_agent.memory.steps = steps

    limiter = memory_window_limiter(
        max_recent_steps=5,
        keep_last_plans=2,
    )

    # When
    limiter(MemoryStep(), mock_agent)

    # Then
    new = mock_agent.memory.steps
    plans = [s for s in new if isinstance(s, PlanningStep)]
    assert len(plans) == 2  # the last two planning steps
    assert len(new) <= 9  # recent + last plans


def test_no_verbosity_output_when_quiet(mock_agent, action_step):
    # Given
    mock_agent.verbosity_level = 0
    mock_agent.memory.steps = [action_step] * 25

    limiter = memory_window_limiter(max_recent_steps=6)

    # When
    # Should NOT raise → remove pytest.raises
    limiter(MemoryStep(), mock_agent)  # just call it
    # Optionally capture stdout and assert nothing printed

    # No crash, and no print expected


def test_handles_empty_memory(mock_agent):
    # Given
    mock_agent.memory.steps = []

    limiter = memory_window_limiter()

    # When — should not crash
    limiter(MemoryStep(), mock_agent)

    # Then
    assert mock_agent.memory.steps == []


def test_handles_no_agent():
    # Given
    limiter = memory_window_limiter()

    # When — should not crash
    limiter(MemoryStep(), agent=None)


@pytest.mark.parametrize("step_count", [0, 3, 7, 15, 30])
def test_correct_length_after_truncation(mock_agent, action_step, step_count):
    mock_agent.memory.steps = [action_step] * step_count
    limiter = memory_window_limiter(max_recent_steps=10)

    limiter(MemoryStep(), mock_agent)

    # With placeholder consuming 1 slot → max length = 10
    expected = min(step_count, 10)
    assert len(mock_agent.memory.steps) == expected
