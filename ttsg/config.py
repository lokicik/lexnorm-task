from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def discover_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing a data directory.")


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    cache: Path
    outputs: Path
    target_only: Path
    final: Path
    benchmark: Path
    deliverables: Path
    tools: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "Paths":
        resolved_root = discover_project_root(root)
        outputs = resolved_root / "outputs"
        return cls(
            root=resolved_root,
            data=resolved_root / "data",
            cache=resolved_root / ".cache",
            outputs=outputs,
            target_only=outputs / "target-only",
            final=outputs / "final",
            benchmark=outputs / "benchmark",
            deliverables=resolved_root / "deliverables",
            tools=resolved_root / "tools",
        )

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.cache,
            self.outputs,
            self.target_only,
            self.final,
            self.benchmark,
            self.deliverables,
            self.tools,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None
    gemini_api_key: str | None
    replicate_api_token: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            replicate_api_token=os.getenv("REPLICATE_API_TOKEN"),
        )
