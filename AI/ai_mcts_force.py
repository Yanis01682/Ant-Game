from __future__ import annotations

import os

try:
    from ai_mcts import MCTSAgent
except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
    if exc.name != "ai_mcts":
        raise
    from AI.ai_mcts import MCTSAgent


class AI(MCTSAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("iterations", 28)
        kwargs.setdefault("max_depth", 3)
        kwargs.setdefault("shortlist_size", 10)
        super().__init__(*args, **kwargs)

    def create_session(self):
        os.environ["AGENT_TRADITION_FORCE_MCTS"] = "1"
        try:
            from protocol import ProtocolSession
        except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
            if exc.name != "protocol":
                raise
            from AI.protocol import ProtocolSession
        return ProtocolSession(self)
