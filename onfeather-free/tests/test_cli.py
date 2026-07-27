"""CLI wiring.

Added after `of-free serve` shipped with a NameError in its parser setup: every
other test drove the library directly, so nothing ever called `main()` and the
typo survived a green suite.
"""

from __future__ import annotations

import pytest

from onfeather_free import cli

SUBCOMMANDS = ["status", "route", "providers", "chat", "serve"]


def parse(argv: list[str]):
    """Parse without executing: the handler is what would run next."""
    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_builds_its_parser(name):
    """Building the parser is what crashed: the NameError fired before argparse
    ever saw the arguments, so every subcommand was equally broken."""
    with pytest.raises(SystemExit) as caught:
        cli.main([name, "--help"])
    assert caught.value.code == 0


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_binds_a_handler(name):
    args = parse([name, "prompt"] if name == "chat" else [name])
    assert callable(getattr(args, "handler", None)), f"{name} has no handler"


def test_top_level_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert "onfeather-free" in capsys.readouterr().out


def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as caught:
        cli.main(["nonsense"])
    assert caught.value.code != 0


def test_missing_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code != 0


def test_serve_defaults_match_the_server_module():
    from onfeather_free import server

    args = parse(["serve"])
    assert args.host == server.DEFAULT_HOST
    assert args.port == server.DEFAULT_PORT


def test_chat_requires_a_prompt():
    with pytest.raises(SystemExit) as caught:
        cli.main(["chat"])
    assert caught.value.code != 0


def test_invalid_strategy_is_rejected():
    with pytest.raises(SystemExit) as caught:
        cli.main(["route", "--strategy", "vibes"])
    assert caught.value.code != 0
