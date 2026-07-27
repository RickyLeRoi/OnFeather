from onfeather_free import config


def test_parses_simple_assignments():
    values = config.parse_env("GROQ_API_KEY=gsk_abc\nGEMINI_API_KEY=aiz_def\n")
    assert values == {"GROQ_API_KEY": "gsk_abc", "GEMINI_API_KEY": "aiz_def"}


def test_ignores_comments_and_blank_lines():
    values = config.parse_env("# a comment\n\nA=1\n   \n#B=2\n")
    assert values == {"A": "1"}


def test_strips_export_prefix():
    """So a file can be both sourced by a shell and read by us."""
    assert config.parse_env("export GROQ_API_KEY=x\n") == {"GROQ_API_KEY": "x"}


def test_strips_matching_quotes():
    values = config.parse_env("A=\"quoted\"\nB='single'\nC=bare\n")
    assert values == {"A": "quoted", "B": "single", "C": "bare"}


def test_keeps_unbalanced_quotes_verbatim():
    assert config.parse_env('A="unclosed\n') == {"A": '"unclosed'}


def test_values_containing_equals_survive():
    """Base64-ish keys with padding would otherwise be truncated."""
    assert config.parse_env("A=abc=def==\n") == {"A": "abc=def=="}


def test_lines_without_equals_are_skipped():
    assert config.parse_env("nonsense\nA=1\n") == {"A": "1"}


def test_load_env_seeds_missing_variables(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=from_file\n")
    environ = {}

    applied = config.load_env(path, environ=environ)

    assert applied == [path]
    assert environ["GROQ_API_KEY"] == "from_file"


def test_exported_variables_win_over_the_file(tmp_path):
    """Overriding a key for one command must not require editing the file."""
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=from_file\n")
    environ = {"GROQ_API_KEY": "from_shell"}

    config.load_env(path, environ=environ)

    assert environ["GROQ_API_KEY"] == "from_shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert config.load_env(tmp_path / "absent", environ={}) == []


def test_redaction_never_reveals_the_middle():
    redacted = config.redact("gsk_1234567890abcdef")
    assert "1234567890" not in redacted
    assert redacted.startswith("gsk_")


def test_redaction_of_short_secrets_reveals_nothing():
    assert config.redact("short") == "*****"


def test_redaction_of_absent_secret():
    assert config.redact(None) == "—"
