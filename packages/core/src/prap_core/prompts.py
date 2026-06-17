from pathlib import Path
from string import Template


class PromptDir:
    """Loads prompt templates from a directory."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"prompt directory not found: {self.dir}")

    def load(self, name: str) -> str:
        path = self.dir / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, name: str, /, **variables: object) -> str:
        return Template(self.load(name)).safe_substitute(**variables)
