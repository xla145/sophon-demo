
基础镜像：registry.cn-hangzhou.aliyuncs.com/xula/tpuc_yolo_onnx

## 前置条件

- 已准备好 `.pt` 模型文件（示例：`/workspace/fgy2.pt`、`/workspace/best_15151515.pt`）
- 建议在容器内执行，避免本机环境依赖冲突

## 进入容器（示例）

```bash
docker run --privileged --name myname --network host -v $PWD:/workspace -it registry.cn-hangzhou.aliyuncs.com/xula/tpuc_yolo_onnx
```

## 转换步骤

```bash
cd /workspace/sample/YOLOv8_plus_det

# fgy2
./scripts/gen_fp16bmodel_from_pt_mlir.sh /workspace/fgy2.pt bm1684x

# best_15151515
./scripts/gen_fp16bmodel_from_pt_mlir.sh /workspace/best_15151515.pt bm1684x
```

## 产物位置

- ONNX：`sample/YOLOv8_plus_det/models/onnx/*.onnx`
- BModel：`sample/YOLOv8_plus_det/models/BM1684X/*_fp16_1b.bmodel`

## 可选参数备忘

脚本用法：

```bash
./scripts/gen_fp16bmodel_from_pt_mlir.sh <pt_path> [target=bm1684x] [model_name=auto] [batch=1]
```

常用环境变量（可选）：

```bash
IMGSZ=640 OPSET=17 DYNAMIC=1 SIMPLIFY=1 ./scripts/gen_fp16bmodel_from_pt_mlir.sh /workspace/fgy2.pt bm1684x
```

## 常见问题

- `ultralytics not installed`：先执行 `pip3 install ultralytics`
- 目标填了 `bm1684`：该目标不支持 `fp16`，请使用 `bm1684x`