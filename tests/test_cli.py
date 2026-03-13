"""Tests for the CLI argument parsing and command dispatch."""

import subprocess
import sys
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISABELLA_CONFIG = os.path.join(PROJECT_ROOT, "configs", "isabella.yaml")


def _run_cli(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"CLI failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


class TestCLIHelp:
    def test_root_help(self):
        result = _run_cli("--help")
        assert "bird-sim" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_paths_help(self):
        result = _run_cli("paths", "--help")
        assert "--config" in result.stdout

    def test_mortality_help(self):
        result = _run_cli("mortality", "--help")
        assert "--config" in result.stdout

    def test_agent_help(self):
        result = _run_cli("agent", "--help")
        assert "--config" in result.stdout

    def test_generate_help(self):
        result = _run_cli("generate", "--help")
        assert "--turbines" in result.stdout

    def test_compare_help(self):
        result = _run_cli("compare", "--help")
        assert "configs" in result.stdout.lower()

    def test_sweep_help(self):
        result = _run_cli("sweep", "--help")
        assert "--param" in result.stdout


class TestCLIInit:
    def test_init_creates_config(self, tmp_path):
        out = str(tmp_path / "new_site.yaml")
        result = _run_cli("init", "--name", "Test Site", "--output", out)
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "Test Site" in content

    def test_init_refuses_overwrite(self, tmp_path):
        out = str(tmp_path / "existing.yaml")
        with open(out, "w") as f:
            f.write("existing")
        result = _run_cli("init", "--name", "X", "--output", out, expect_ok=False)
        assert result.returncode != 0


class TestCLIErrorHandling:
    def test_no_command_shows_help(self):
        result = _run_cli(expect_ok=False)
        assert result.returncode != 0

    def test_missing_config_errors(self):
        result = _run_cli("mortality", "--config", "/nonexistent/path.yaml", expect_ok=False)
        assert result.returncode != 0


class TestCLINamesStripping:
    """Verify that --names argument strips whitespace."""

    def test_names_strip(self):
        from src.cli import cmd_compare
        import argparse
        args = argparse.Namespace(
            configs=[ISABELLA_CONFIG, ISABELLA_CONFIG],
            names="  Site A , Site B  ",
            out=None,
        )
        names = [n.strip() for n in args.names.split(",")] if args.names else None
        assert names == ["Site A", "Site B"]
