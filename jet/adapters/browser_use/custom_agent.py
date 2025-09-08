from pathlib import Path
from typing import Optional

from browser_use import Agent
from browser_use.agent.views import BrowserStateHistory, StepMetadata
from browser_use.browser.views import BrowserStateSummary


class CustomAgent(Agent):
    """Custom Agent that allows specifying a custom directory for screenshots and other files."""

    def __init__(self, *args, custom_screenshot_dir: str | Path | None = None, **kwargs):
        """
        Initialize the CustomAgent with an optional custom screenshot directory.

        Args:
            custom_screenshot_dir (str | Path | None): Directory to store screenshots and agent files.
            *args: Positional arguments passed to the base Agent class.
            **kwargs: Keyword arguments passed to the base Agent class.
        """
        super().__init__(*args, **kwargs)
        if custom_screenshot_dir:
            self.agent_directory = Path(custom_screenshot_dir).resolve()
        # Re-initialize screenshot service with custom directory
        self._set_screenshot_service()

    async def _finalize(self, browser_state_summary: BrowserStateSummary | None) -> None:
        """
        Finalize the step by saving history, logging, emitting events, and capturing an end-of-step screenshot.

        Args:
            browser_state_summary (BrowserStateSummary | None): The browser state summary from the step start.
        """
        import time
        step_end_time = time.time()
        if not self.state.last_result:
            return

        if browser_state_summary:
            metadata = StepMetadata(
                step_number=self.state.n_steps,
                step_start_time=self.step_start_time,
                step_end_time=step_end_time,
            )

            # Save start-of-step screenshot and history
            await self._make_history_item(
                self.state.last_model_output,
                browser_state_summary,
                self.state.last_result,
                metadata
            )

            # Capture and store end-of-step screenshot
            end_browser_state = await self.browser_session.get_browser_state_summary(
                cache_clickable_elements_hashes=False,  # No need to cache at step end
                include_screenshot=True,  # Capture screenshot
                include_recent_events=False,  # Not needed for screenshot
            )
            end_screenshot_path = None
            if end_browser_state.screenshot:
                self.logger.debug(
                    f'📸 Storing end-of-step screenshot for step {self.state.n_steps}')
                end_screenshot_path = await self.screenshot_service.store_screenshot(
                    end_browser_state.screenshot, f"step_{self.state.n_steps}_end"
                )
                self.logger.debug(
                    f'📸 End-of-step screenshot stored at: {end_screenshot_path}')

            # Update history with end-of-step screenshot path (optional, requires BrowserStateHistory extension)
            if end_screenshot_path and self.history.history:
                last_history_item = self.history.history[-1]
                if hasattr(last_history_item.state, 'end_screenshot_path'):
                    last_history_item.state.end_screenshot_path = end_screenshot_path

        # Log step completion summary
        self._log_step_completion_summary(
            self.step_start_time, self.state.last_result)

        # Save file system state after step completion
        self.save_file_system_state()

        # Emit both step created and executed events
        if browser_state_summary and self.state.last_model_output:
            actions_data = []
            if self.state.last_model_output.action:
                for action in self.state.last_model_output.action:
                    action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
                    actions_data.append(action_dict)

            step_event = self.CreateAgentStepEvent.from_agent_step(
                self,
                self.state.last_model_output,
                self.state.last_result,
                actions_data,
                browser_state_summary,
            )
            self.eventbus.dispatch(step_event)

        # Increment step counter after step is fully completed
        self.state.n_steps += 1
