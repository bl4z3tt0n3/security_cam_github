"""Explicitly export the official FaceNet 20180402-114759 graph to ONNX.

This utility is intentionally offline: it accepts an already downloaded and
verified checkpoint directory and never downloads weights or installs Python
packages.  TensorFlow and tf2onnx must be supplied by an operator-managed
isolated environment.  The exported graph is inference-only, fixes
``phase_train`` to ``False`` and exposes the NCHW input expected by the local
face embedding adapter.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import os
import re
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SCRIPT_ROOT / "models/face_embedding/facenet-20180402-vggface2.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_training_flag(graph_def, *, tensorflow) -> None:
    for node in graph_def.node:
        if node.name != "phase_train":
            continue
        node.ClearField("input")
        node.ClearField("attr")
        node.op = "Const"
        node.attr["dtype"].type = tensorflow.bool.as_datatype_enum
        node.attr["value"].tensor.CopyFrom(
            tensorflow.make_tensor_proto(False, dtype=tensorflow.bool)
        )
        return
    raise ValueError("official FaceNet graph does not contain phase_train")


def _canonicalize_onnx(path: Path, *, onnx) -> None:
    """Remove tf2onnx run-local numeric suffixes before hashing the export."""

    model = onnx.load(str(path))
    suffixes: dict[str, str] = {}
    pattern = re.compile(r"__\d+")

    def canonical(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token not in suffixes:
                suffixes[token] = f"__{len(suffixes) + 1}"
            return suffixes[token]

        return pattern.sub(replace, value)

    def visit(message) -> None:
        for field, value in message.ListFields():
            if field.type == field.TYPE_STRING:
                values = getattr(message, field.name)
                if field.is_repeated:
                    for index in range(len(values)):
                        values[index] = canonical(values[index])
                else:
                    setattr(message, field.name, canonical(value))
            elif field.type == field.TYPE_MESSAGE:
                if field.is_repeated:
                    for child in value:
                        visit(child)
                else:
                    visit(value)

    visit(model)
    initializers_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    stable_constants: dict[str, str] = {}
    for node in model.graph.node:
        for source in node.input:
            if source not in initializers_by_name or not source.endswith("/mul/x:0"):
                continue
            stable_constants.setdefault(
                source,
                re.sub(r"/(block\d+)_\d+/mul/x:0$", r"/\1/mul/constant_x", source),
            )
    for source, stable_name in stable_constants.items():
        initializers_by_name[source].name = stable_name
    for node in model.graph.node:
        for index, source in enumerate(node.input):
            if source in stable_constants:
                node.input[index] = stable_constants[source]
    initializers = sorted(model.graph.initializer, key=lambda value: value.name)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    canonical_path = path.with_name(path.name + ".canonical")
    canonical_path.unlink(missing_ok=True)
    onnx.save(model, str(canonical_path))
    os.replace(canonical_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the local official FaceNet 20180402-114759 graph to ONNX."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing the official 20180402-114759.pb file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination ONNX path; defaults to the registered FaceNet artifact.",
    )
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output artifact explicitly.",
    )
    args = parser.parse_args(argv)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    source = checkpoint_dir / "20180402-114759.pb"
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"official FaceNet graph not found: {source}")
    if output.exists() and not args.force:
        parser.error(f"output exists; pass --force to replace it: {output}")

    try:
        import tensorflow as tf
        import onnx
        import tf2onnx
    except ImportError as exc:
        print(
            "NOT READY: TensorFlow and tf2onnx are required in an isolated export environment",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    tf.compat.v1.disable_eager_execution()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(source.read_bytes())
    _freeze_training_flag(graph_def, tensorflow=tf)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    temporary.unlink(missing_ok=True)
    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=["input:0"],
        output_names=["embeddings:0"],
        inputs_as_nchw=["input:0"],
        shape_override={"input:0": [None, 160, 160, 3]},
        opset=args.opset,
        output_path=str(temporary),
    )
    _canonicalize_onnx(temporary, onnx=onnx)
    os.replace(temporary, output)
    print(f"SOURCE {source} sha256={_sha256(source)}")
    print(f"READY {output} sha256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
