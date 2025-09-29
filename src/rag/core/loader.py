from pathlib import Path


class TextLoader:
    """Loads plain text data from a local file.

    Keeps the interface simple and explicit for the single-file use case.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> str:
        """Read the entire file content as a single string.

        Returns:
            The full text content of the file.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        return self.file_path.read_text(encoding="utf-8")
