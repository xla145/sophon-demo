#!/bin/bash
set -euo pipefail

model_dir=$(dirname $(readlink -f "$0"))

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  $0 <pt_path> [target=bm1684x] [model_name=auto] [batch=1]"
  echo ""
  echo "Env vars (optional):"
  echo "  IMGSZ=640 OPSET=17 DYNAMIC=1 SIMPLIFY=1"
  exit 1
fi

pt_path="$1"
target="${2:-bm1684x}"
target="${target,,}"
target_dir="${target^^}"

model_name="${3:-}"
batch="${4:-1}"

IMGSZ="${IMGSZ:-640}"
OPSET="${OPSET:-17}"
DYNAMIC="${DYNAMIC:-1}"
SIMPLIFY="${SIMPLIFY:-1}"
AUTO_PATCH_ULTRALYTICS="${AUTO_PATCH_ULTRALYTICS:-1}"

if test "$target" = "bm1684"; then
  echo "bm1684 do not support fp16"
  exit 1
fi

if [ -z "$model_name" ]; then
  base="$(basename "$pt_path")"
  model_name="${base%.*}"
fi

outdir="$model_dir/../models/$target_dir"
onnx_dir="$model_dir/../models/onnx"
onnx_path="$onnx_dir/${model_name}.onnx"

function patch_ultralytics_predict_once()
{
  if [ "$AUTO_PATCH_ULTRALYTICS" != "1" ]; then
    echo "[PATCH] skip ultralytics patch (AUTO_PATCH_ULTRALYTICS=$AUTO_PATCH_ULTRALYTICS)"
    return 0
  fi

  python3 - <<'PY'
import importlib.util
import os
import re
import sys

spec = importlib.util.find_spec("ultralytics")
if spec is None or not spec.origin:
    print("ERROR: ultralytics not installed. Please install it first.", file=sys.stderr)
    print("  pip3 install ultralytics", file=sys.stderr)
    sys.exit(1)

pkg_root = os.path.dirname(spec.origin)
tasks_path = os.path.join(pkg_root, "nn", "tasks.py")
if not os.path.isfile(tasks_path):
    print(f"ERROR: tasks.py not found: {tasks_path}", file=sys.stderr)
    sys.exit(1)

with open(tasks_path, "r", encoding="utf-8") as f:
    content = f.read()
    lines = content.splitlines(keepends=True)

if "return x.permute(0, 2, 1)" in content:
    print(f"[PATCH] already patched: {tasks_path}")
    sys.exit(0)

# Find function body safely and preserve original indentation.
def_line_idx = -1
for i, line in enumerate(lines):
    if re.match(r"^\s*def _predict_once\(self, x, profile=False, visualize=False, embed=None\):\s*$", line):
        def_line_idx = i
        break

if def_line_idx < 0:
    print("ERROR: cannot locate `_predict_once` function in tasks.py.", file=sys.stderr)
    sys.exit(1)

def_indent = len(lines[def_line_idx]) - len(lines[def_line_idx].lstrip(" \t"))
ret_line_idx = -1
ret_indent = ""

for i in range(def_line_idx + 1, len(lines)):
    raw = lines[i]
    stripped = raw.strip()

    if stripped == "":
        continue

    curr_indent = len(raw) - len(raw.lstrip(" \t"))
    if curr_indent <= def_indent and not stripped.startswith("#"):
        break

    if stripped == "return x":
        ret_line_idx = i
        ret_indent = raw[: len(raw) - len(raw.lstrip(" \t"))]
        break

if ret_line_idx < 0:
    print("ERROR: cannot locate `return x` inside `_predict_once`.", file=sys.stderr)
    print("Please patch manually according to docs/YOLOv8_Export_Guide.md", file=sys.stderr)
    sys.exit(1)

line_ending = "\n" if lines[ret_line_idx].endswith("\n") else ""
lines[ret_line_idx] = f"{ret_indent}return x.permute(0, 2, 1){line_ending}"
new_content = "".join(lines)
backup_path = tasks_path + ".bak_for_tpu_mlir"
if not os.path.exists(backup_path):
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)

with open(tasks_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"[PATCH] success: {tasks_path}")
print(f"[PATCH] backup: {backup_path}")
PY
}

function export_onnx_from_pt()
{
  mkdir -p "$onnx_dir"
  echo "[PT->ONNX] exporting: $pt_path -> $onnx_path"
  exported_onnx="$(
    python3 - "$pt_path" "$batch" "$IMGSZ" "$OPSET" "$DYNAMIC" "$SIMPLIFY" <<'PY'
import os
import sys

pt_path = sys.argv[1]
batch = int(sys.argv[2])
imgsz = int(sys.argv[3])
opset = int(sys.argv[4])
dynamic = bool(int(sys.argv[5]))
simplify = bool(int(sys.argv[6]))

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics not available. Please install it first:", file=sys.stderr)
    print("  pip3 install ultralytics", file=sys.stderr)
    print("Then refer to docs/YOLOv8_Export_Guide.md for export notes.", file=sys.stderr)
    raise

model = YOLO(pt_path)
exported = model.export(format="onnx", opset=opset, dynamic=dynamic, simplify=simplify, batch=batch, imgsz=imgsz)

if isinstance(exported, (list, tuple)):
    exported = exported[0] if exported else ""

exported = str(exported) if exported is not None else ""
if not exported:
    # Fallback: try to infer export path beside pt
    exported = os.path.splitext(pt_path)[0] + ".onnx"

print("__ONNX_PATH__=" + exported)
PY
  )"

  exported_onnx="${exported_onnx##*__ONNX_PATH__=}"
  if [ ! -f "$exported_onnx" ]; then
    echo "ERROR: exported onnx not found: $exported_onnx"
    exit 1
  fi

  cp -f "$exported_onnx" "$onnx_path"
}

function gen_mlir()
{
  model_transform.py \
    --model_name "${model_name}" \
    --model_def "$onnx_path" \
    --input_shapes "[[${batch},3,${IMGSZ},${IMGSZ}]]" \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --mlir "${model_name}_${batch}b.mlir"
}

function gen_fp16bmodel()
{
  model_deploy.py \
    --mlir "${model_name}_${batch}b.mlir" \
    --quantize F16 \
    --chip "$target" \
    --model "${model_name}_fp16_${batch}b.bmodel"

  mv "${model_name}_fp16_${batch}b.bmodel" "$outdir/"

  if test "$target" = "bm1688"; then
    model_deploy.py \
      --mlir "${model_name}_${batch}b.mlir" \
      --quantize F16 \
      --chip "$target" \
      --model "${model_name}_fp16_${batch}b_2core.bmodel" \
      --num_core 2

    mv "${model_name}_fp16_${batch}b_2core.bmodel" "$outdir/"
  fi
}

pushd "$model_dir" >/dev/null

if [ ! -f "$pt_path" ]; then
  echo "ERROR: pt file not found: $pt_path"
  exit 1
fi

mkdir -p "$outdir"
patch_ultralytics_predict_once
export_onnx_from_pt
gen_mlir
gen_fp16bmodel

popd >/dev/null

