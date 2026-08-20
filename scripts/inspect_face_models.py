"""Read-only inspection of registered local face-model inputs and outputs.

This command never downloads, converts, exports or writes model files.  It is
kept separate from the normal test suite so missing optional recognizers remain
an explicit operational result instead of a falsely green test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.face.registry import (  # noqa: E402
    FACE_DETECTOR_SPECS,
    LANDMARKER_SPEC,
    RECOGNIZER_SPECS,
    FaceModelSpec,
    RecognizerSpec,
    model_path,
)


def _id(spec: FaceModelSpec | RecognizerSpec) -> str:
    return spec.model_id if isinstance(spec, FaceModelSpec) else spec.recognizer_id


def _all_specs() -> tuple[FaceModelSpec | RecognizerSpec, ...]:
    return (*FACE_DETECTOR_SPECS, LANDMARKER_SPEC, *RECOGNIZER_SPECS)


def _inspect(spec: FaceModelSpec | RecognizerSpec, root: Path) -> str:
    path = model_path(spec, root)
    if path.suffix.lower() == ".xml":
        binary = path.with_suffix(".bin")
        if not path.is_file() or not binary.is_file():
            return f"MISSING xml/bin: {path}"
        try:
            from openvino import Core

            model = Core().read_model(str(path))
            inputs = [
                f"{port.any_name}:{tuple(port.partial_shape)}" for port in model.inputs
            ]
            outputs = [
                f"{port.any_name}:{tuple(port.partial_shape)}" for port in model.outputs
            ]
            return f"READY inputs={inputs} outputs={outputs}"
        except Exception as exc:
            return f"INSPECTION_ERROR {type(exc).__name__}: {exc}"
    if not path.is_file():
        return f"MISSING: {path}"
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = [f"{item.name}:{tuple(item.shape)}:{item.type}" for item in session.get_inputs()]
        outputs = [f"{item.name}:{tuple(item.shape)}:{item.type}" for item in session.get_outputs()]
        providers = tuple(session.get_providers())
        return f"READY providers={providers} inputs={inputs} outputs={outputs}"
    except Exception as exc:
        return f"INSPECTION_ERROR {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local face-model I/O without writing files.")
    parser.add_argument("--model", action="append", dest="models", help="Registered model id; repeatable.")
    parser.add_argument("--all", action="store_true", help="Inspect every registered model.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root for relative model paths.")
    args = parser.parse_args(argv)
    selected = _all_specs() if args.all or not args.models else tuple(
        spec for spec in _all_specs() if _id(spec) in set(args.models)
    )
    if args.models:
        known = {_id(spec) for spec in _all_specs()}
        unknown = sorted(set(args.models) - known)
        if unknown:
            parser.error("unknown model id: " + ", ".join(unknown))
    root = args.root.expanduser().resolve()
    exit_code = 0
    for spec in selected:
        result = _inspect(spec, root)
        print(f"{_id(spec)}: {result}")
        if not result.startswith("READY"):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
