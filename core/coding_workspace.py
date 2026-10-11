import difflib
import os
import re
import stat
from pathlib import Path


CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".dart", ".ex", ".exs", ".go", ".h",
    ".hpp", ".hrl", ".html", ".java", ".js", ".json", ".jsx", ".kt",
    ".kts", ".lua", ".md", ".php", ".pl", ".pm", ".py", ".r", ".rb",
    ".rs", ".scala", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx",
    ".txt", ".vue", ".xml", ".yaml", ".yml",
}
MAX_SOURCE_BYTES = 128 * 1024
BLOCKED_NAMES = re.compile(
    r"(^|[._-])(.env|secrets?|credentials?|passwords?|tokens?|private[_-]?keys?|api[_-]?keys?)"
    r"([._-]|$)",
    re.IGNORECASE,
)


class CodingWorkspace:
    def __init__(self, root_directory=None):
        self.root = None
        if root_directory:
            root = Path(root_directory).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError(f"LUCY_CODE_DIR must be a directory: {root}")
            self.root = root

    def _resolve(self, relative_path):
        if self.root is None:
            raise RuntimeError(
                "Coding access is disabled. Set LUCY_CODE_DIR to the project folder Lucy may edit."
            )
        normalized = relative_path.replace("\\", "/").strip()
        relative = Path(normalized)
        if (
            not normalized
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or any(part.startswith(".") for part in relative.parts)
        ):
            raise ValueError("Use a visible relative file path inside LUCY_CODE_DIR.")
        if any(BLOCKED_NAMES.search(part) for part in relative.parts):
            raise ValueError("Files that may contain secrets or credentials are not accessible.")
        if relative.suffix.lower() not in CODE_SUFFIXES:
            raise ValueError("That file type is not available to Lucy's coding tools.")

        candidate = self.root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("Symbolic links are not accessible.")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("That path is outside LUCY_CODE_DIR.") from error
        if resolved.exists() and not resolved.is_file():
            raise ValueError("The selected path is not a regular file.")
        if resolved.exists() and resolved.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"Files must be {MAX_SOURCE_BYTES // 1024} KB or smaller.")
        if not resolved.parent.is_dir():
            raise ValueError("The parent directory must already exist.")
        return resolved

    def list_files(self, relative_directory=""):
        if self.root is None:
            raise RuntimeError(
                "Coding access is disabled. Set LUCY_CODE_DIR to the project folder Lucy may edit."
            )
        if relative_directory.strip() in ("", "."):
            folder = self.root
        else:
            normalized = relative_directory.replace("\\", "/").strip()
            relative = Path(normalized)
            if relative.is_absolute() or any(
                part in ("", ".", "..") or part.startswith(".") for part in relative.parts
            ):
                raise ValueError("Use a visible relative folder path inside LUCY_CODE_DIR.")
            folder = self.root
            for part in relative.parts:
                folder = folder / part
                if folder.is_symlink():
                    raise ValueError("Symbolic links are not accessible.")
            folder = folder.resolve(strict=True)
            try:
                folder.relative_to(self.root)
            except ValueError as error:
                raise ValueError("That folder is outside LUCY_CODE_DIR.") from error
            if not folder.is_dir():
                raise ValueError("That path is not a folder.")

        files = []
        for path in folder.iterdir():
            if path.name.startswith(".") or path.is_symlink():
                continue
            if (
                path.is_file()
                and path.suffix.lower() in CODE_SUFFIXES
                and not BLOCKED_NAMES.search(path.name)
                and path.stat().st_size <= MAX_SOURCE_BYTES
            ):
                files.append(path.relative_to(self.root).as_posix())
        return sorted(files)[:200]

    def read_file(self, relative_path):
        path = self._resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {relative_path}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Only UTF-8 text files are available to Lucy.") from error
        return content

    def preview_and_write(self, relative_path, new_content, approval_callback=None):
        if not isinstance(new_content, str):
            raise ValueError("Proposed file contents must be text.")
        encoded = new_content.encode("utf-8")
        if len(encoded) > MAX_SOURCE_BYTES:
            raise ValueError(f"Proposed files must be {MAX_SOURCE_BYTES // 1024} KB or smaller.")
        path = self._resolve(relative_path)
        relative_name = path.relative_to(self.root).as_posix()
        old_content = self.read_file(relative_path) if path.exists() else ""
        if old_content == new_content:
            return f"No change needed in {relative_name}."

        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{relative_name}" if path.exists() else "/dev/null",
                tofile=f"b/{relative_name}",
            )
        )
        if not diff:
            diff = f"Replace the contents of {relative_name} ({len(encoded)} bytes)."
        if approval_callback is None or not approval_callback(
            f"Apply this proposed code change?\n{diff}"
        ):
            return f"Change to {relative_name} was not applied."

        temporary_path = path.with_name(path.name + ".lucy-tmp")
        if temporary_path.exists() or temporary_path.is_symlink():
            raise ValueError("A temporary file already exists at the write location.")
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        try:
            with temporary_path.open("x", encoding="utf-8") as file:
                file.write(new_content)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temporary_path, mode)
            temporary_path.replace(path)
        except OSError:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
        return f"Applied your approved change to {relative_name}."

    def check_python_syntax(self, relative_path):
        path = self._resolve(relative_path)
        if path.suffix.lower() != ".py":
            raise ValueError("Syntax checking is only available for Python files.")
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {relative_path}")
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except UnicodeDecodeError as error:
            raise ValueError("Python source must be UTF-8.") from error
        except SyntaxError as error:
            return (
                f"Python syntax error in {relative_path}, line {error.lineno}: "
                f"{error.msg}"
            )
        return f"Python syntax is valid in {relative_path}. Lucy compiled it without executing it."

    def delete_file(self, relative_path, approval_callback=None):
        path = self._resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {relative_path}")
        content = self.read_file(relative_path)
        relative_name = path.relative_to(self.root).as_posix()
        preview = f"Move this file to Lucy's recoverable trash?\n--- a/{relative_name}\n"
        preview += "".join(f"-{line}\n" for line in content.splitlines())
        if approval_callback is None or not approval_callback(preview):
            return f"Removal of {relative_name} was not approved."

        trash_root = self.root / ".lucy-trash"
        destination = trash_root / relative_name
        if trash_root.is_symlink():
            raise ValueError("The coding workspace trash folder must not be a symbolic link.")
        current = self.root
        for part in (".lucy-trash",) + Path(relative_name).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symbolic links are not accessible.")
            if not current.exists():
                current.mkdir()
        if destination.exists():
            raise FileExistsError("A trashed file with that path already exists.")
        path.replace(destination)
        return f"Moved {relative_name} to recoverable trash. Use /code-restore {relative_name} to restore it."

    def restore_file(self, relative_path, approval_callback=None):
        path = self._resolve(relative_path)
        if path.exists():
            raise FileExistsError(f"A file already exists at {relative_path}.")
        trash_root = self.root / ".lucy-trash"
        if trash_root.is_symlink():
            raise ValueError("The coding workspace trash folder must not be a symbolic link.")
        trashed_path = trash_root / Path(relative_path.replace("\\", "/").strip())
        current = trash_root
        for part in Path(relative_path.replace("\\", "/").strip()).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symbolic links are not accessible.")
        resolved_trash = trashed_path.resolve(strict=True)
        try:
            resolved_trash.relative_to(trash_root.resolve(strict=True))
        except ValueError as error:
            raise ValueError("That trashed file is outside Lucy's recoverable trash.") from error
        if not resolved_trash.is_file() or resolved_trash.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("That trashed item is not a supported source file.")
        preview = (
            f"Restore {relative_path} from Lucy's recoverable trash?\n"
            f"--- /dev/null\n+++ b/{relative_path}\n"
        )
        if approval_callback is None or not approval_callback(preview):
            return f"Restore of {relative_path} was not approved."
        resolved_trash.replace(path)
        return f"Restored {relative_path}."
