"""Tests for sandbox isolation — OS-level enforcement via sandbox-exec."""

import os

import pytest

from core.tools import ToolRegistry, set_context
from tools.builtins import register


@pytest.fixture
def registry():
    r = ToolRegistry()
    register(r, sandbox_dir="./sandbox")
    set_context({"chat_id": "test-sandbox"})
    yield r
    # Cleanup
    import shutil

    shutil.rmtree("./sandbox/test-sandbox", ignore_errors=True)


class TestSandboxIsolation:
    async def test_python_runs(self, registry):
        r = await registry.execute("run_code", {"code": "print(1+1)", "language": "python"})
        assert "2" in r

    async def test_bash_runs(self, registry):
        r = await registry.execute("run_code", {"code": "echo hello", "language": "bash"})
        assert "hello" in r

    async def test_write_inside_sandbox(self, registry):
        r = await registry.execute("run_code", {"code": "echo ok > test.txt && cat test.txt", "language": "bash"})
        assert "ok" in r

    async def test_write_outside_blocked(self, registry):
        r = await registry.execute("run_code", {"code": "echo pwned > /tmp/memoo-escape-test.txt", "language": "bash"})
        assert "Operation not permitted" in r or "Permission denied" in r
        assert not os.path.exists("/tmp/memoo-escape-test.txt")

    async def test_python_open_outside_blocked(self, registry):
        r = await registry.execute(
            "run_code", {"code": 'open("/tmp/memoo-py-escape.txt", "w").write("pwned")', "language": "python"}
        )
        assert "error" in r.lower() or "denied" in r.lower() or "not permitted" in r.lower()

    async def test_files_persist_across_calls(self, registry):
        await registry.execute("run_code", {"code": "echo persist > keep.txt", "language": "bash"})
        r = await registry.execute("run_code", {"code": "cat keep.txt", "language": "bash"})
        assert "persist" in r

    async def test_session_isolation(self, registry):
        # Write in session A
        set_context({"chat_id": "session-a"})
        await registry.execute("run_code", {"code": "echo secret > data.txt", "language": "bash"})

        # Session B cannot see it
        set_context({"chat_id": "session-b"})
        r = await registry.execute("run_code", {"code": "cat data.txt 2>&1 || echo NOTFOUND", "language": "bash"})
        assert "NOTFOUND" in r or "No such file" in r

        # Cleanup
        import shutil

        shutil.rmtree("./sandbox/session-a", ignore_errors=True)
        shutil.rmtree("./sandbox/session-b", ignore_errors=True)

    async def test_network_allowed(self, registry):
        r = await registry.execute(
            "run_code",
            {"code": 'curl -s -o /dev/null -w "%{http_code}" https://example.com', "language": "bash"},
        )
        assert "200" in r

    async def test_python_in_bash_allowed(self, registry):
        r = await registry.execute("run_code", {"code": 'python3 -c "print(42)"', "language": "bash"})
        assert "42" in r

    async def test_read_file_path_traversal(self, registry):
        r = await registry.execute("read_file", {"path": "../../config.yaml"})
        assert "escapes sandbox" in r.lower() or "error" in r.lower()

    async def test_timeout(self, registry):
        r = await registry.execute("run_code", {"code": "sleep 10", "language": "bash", "timeout": 2})
        assert "timed out" in r.lower()
