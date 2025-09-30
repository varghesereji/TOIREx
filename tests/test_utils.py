import pytest
from pathlib import Path

from toirex.utils import read_txt_file


@pytest.fixture
def tmp_text_file(tmp_path: Path):
    """Creates a temporaty text file and returns its path."""
    content = [
        "123 abc def\n",
        "456 ghi jkl\n",
        "789 mno\n"
        ]

    file_path = tmp_path / "test_file.txt"
    file_path.write_text("".join(content))
    return file_path


def test_read_txt_file(tmp_text_file: Path):
    expected = [
        ["123", "abc", "def"],
        ["456", "ghi", "jkl"],
        ["789", "mno"]
        ]
    assert read_txt_file(str(tmp_text_file)) == expected


def test_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert read_txt_file(str(empty_file)) == []


def test_whitespace_lines(tmp_path: Path):
    content = [
        "   a b c   \n",
        "\n",
        "  d e  f \n"
    ]
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("".join(content))

    expected = [
        ["a", "b", "c"],
        [""],
        ["d", "e", "", "f"]
    ]
    assert read_txt_file(str(file_path)) == expected
