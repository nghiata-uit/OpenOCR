## 🇻🇳 Vietnamese Fine-Tuning & Inference on VinText (SVTRv2)

We provide a complete pipeline for fine-tuning the SVTRv2 model to detect and recognize Vietnamese text using the [VinText dataset](https://github.com/VinAIResearch/dict-guided).

### 1. Environment & Dataset Preparation
First, set up your environment as described in [Environment](#2-environment).
Then, prepare your VinText dataset and update the dataset paths (`data_dir` and `label_file_list`) in `configs/rec/svtrv2/svtrv2_vn.yml` accordingly.

### 2. Fine-tuning (Training)
Start fine-tuning from our pre-trained SVTRv2 model. You can adjust the batch size and learning rate based on your hardware.

```shell
# For Single GPU
python tools/train_rec.py --c configs/rec/svtrv2/svtrv2_vn.yml

# For Multi-GPU (e.g., 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_vn.yml
```

### 3. High-Speed Inference (Latency Measurement)
For real-time applications, we provide an optimized inference config (`svtrv2_vn_infer.yml`) that bypasses the autoregressive GTC decoder and utilizes the ultra-fast CTC head while maintaining high accuracy.

```shell
# Run fast inference on an image or folder
python tools/infer_rec.py \
    --c configs/rec/svtrv2/svtrv2_vn_infer.yml \
    --o Global.infer_img=/path/to/your/vietnamese_images \
        Global.pretrained_model=./output/rec/vn/svtrv2_vn/best.pth
```