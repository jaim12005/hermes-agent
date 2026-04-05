import types
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider


class TestHonchoProviderAutoInjection:
    def test_prefetch_returns_empty_even_with_cached_dialectic_result(self):
        provider = HonchoMemoryProvider()
        provider._cron_skipped = False
        provider._recall_mode = "hybrid"
        provider._turn_count = 0
        provider._prefetch_result = "stale unrelated project summary"
        provider._config = types.SimpleNamespace(context_tokens=100)

        assert provider.prefetch("what next?") == ""
        assert provider._prefetch_result == "stale unrelated project summary"

    def test_queue_prefetch_refreshes_context_only(self):
        provider = HonchoMemoryProvider()
        provider._cron_skipped = False
        provider._manager = MagicMock()
        provider._session_key = "telegram:dm:test"
        provider._recall_mode = "hybrid"
        provider._context_cadence = 1
        provider._turn_count = 3
        provider._last_context_turn = -999

        provider.queue_prefetch("status?")

        provider._manager.prefetch_context.assert_called_once_with("telegram:dm:test", "status?")
        provider._manager.dialectic_query.assert_not_called()

    def test_queue_prefetch_noops_in_tools_mode(self):
        provider = HonchoMemoryProvider()
        provider._cron_skipped = False
        provider._manager = MagicMock()
        provider._session_key = "telegram:dm:test"
        provider._recall_mode = "tools"

        provider.queue_prefetch("status?")

        provider._manager.prefetch_context.assert_not_called()
        provider._manager.dialectic_query.assert_not_called()
