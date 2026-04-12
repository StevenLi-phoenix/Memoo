"""Tests for cron field validation."""

from datetime import datetime

from core.scheduler import _match_cron_field, cron_matches


class TestCronValidation:
    def test_wildcard(self):
        assert _match_cron_field("*", 5) is True

    def test_exact(self):
        assert _match_cron_field("5", 5) is True
        assert _match_cron_field("5", 6) is False

    def test_step(self):
        assert _match_cron_field("*/5", 10) is True
        assert _match_cron_field("*/5", 7) is False

    def test_step_zero(self):
        # */0 should not crash
        assert _match_cron_field("*/0", 5) is False

    def test_range(self):
        assert _match_cron_field("1-5", 3) is True
        assert _match_cron_field("1-5", 6) is False

    def test_list(self):
        assert _match_cron_field("1,3,5", 3) is True
        assert _match_cron_field("1,3,5", 4) is False

    def test_malformed(self):
        assert _match_cron_field("abc", 5) is False
        assert _match_cron_field("", 5) is False

    def test_cron_matches(self):
        dt = datetime(2026, 4, 12, 8, 30)  # Saturday
        assert cron_matches("30 8 * * *", dt) is True
        assert cron_matches("0 8 * * *", dt) is False

    def test_cron_wrong_fields(self):
        dt = datetime(2026, 4, 12, 8, 30)
        assert cron_matches("only three fields", dt) is False
