# Technical Report

| Field | Value |
|---|---|
| Model | YOLOv8n-pose |
| Runtime | NCNN (ARM CPU) |
| Quantization/optimization | Exported .pt -> NCNN format for ARM-native inference |
| Model size | 6,622 KB (~6.6 MB) |
| Inference latency | 15+ FPS |
| CPU usage | Moderate |
| GPU/NPU usage | None used |
| Tested device | Raspberry Pi 5, 8GB RAM |