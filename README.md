## 🇻🇳 Vietnamese Fine-Tuning & Inference on VinText (SVTRv2)

We provide a complete pipeline for fine-tuning the SVTRv2 model to detect and recognize Vietnamese text using the VinText dataset.

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

## Reproduction schedule:

### Scene Text Recognition

| Method                                        | Venue                                                                                          | Training | Evaluation | Contributor                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------- | ---------- | ------------------------------------------- |
| [CRNN](./configs/rec/svtrs/)                  | [TPAMI 2016](https://arxiv.org/abs/1507.05717)                                                 | ✅       | ✅         |                                             |
| [ASTER](./configs/rec/aster/)                 | [TPAMI 2019](https://ieeexplore.ieee.org/document/8395027)                                     | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [NRTR](./configs/rec/nrtr/)                   | [ICDAR 2019](https://arxiv.org/abs/1806.00926)                                                 | ✅       | ✅         |                                             |
| [SAR](./configs/rec/sar/)                     | [AAAI 2019](https://aaai.org/papers/08610-show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition/) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [MORAN](./configs/rec/moran/)                 | [PR 2019](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300263)             | ✅       | ✅         |                                             |
| [DAN](./configs/rec/dan/)                     | [AAAI 2020](https://arxiv.org/pdf/1912.10205)                                                  | ✅       | ✅         |                                             |
| [RobustScanner](./configs/rec/robustscanner/) | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3160_ECCV_2020_paper.php)   | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [AutoSTR](./configs/rec/autostr/)             | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690732.pdf)            | ✅       | ✅         |                                             |
| [SRN](./configs/rec/srn/)                     | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html) | ✅       | ✅         | [pretto0](https://github.com/pretto0)       |
| [SEED](./configs/rec/seed/)                   | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html) | ✅       | ✅         |                                             |
| [ABINet](./configs/rec/abinet/)               | [CVPR 2021](https://openaccess.thecvf.com//content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [VisionLAN](./configs/rec/visionlan/)         | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html) | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| PIMNet                                        | [ACM MM 2021](https://dl.acm.org/doi/10.1145/3474085.3475238)                                  |          |            | TODO                                        |
| [SVTR](./configs/rec/svtrs/)                  | [IJCAI 2022](https://www.ijcai.org/proceedings/2022/124)                                       | ✅       | ✅         |                                             |
| [PARSeq](./configs/rec/parseq/)               | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)            | ✅       | ✅         |                                             |
| [MATRN](./configs/rec/matrn/)                 | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)            | ✅       | ✅         |                                             |
| [MGP-STR](./configs/rec/mgpstr/)              | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)            | ✅       | ✅         |                                             |
| [LPV](./configs/rec/lpv/)                     | [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0189.pdf)                                  | ✅       | ✅         |                                             |
| [MAERec](./configs/rec/maerec/)(Union14M)     | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [LISTER](./configs/rec/lister/)               | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.pdf) | ✅       | ✅         |                                             |
| [CDistNet](./configs/rec/cdistnet/)           | [IJCV 2024](https://link.springer.com/article/10.1007/s11263-023-01880-0)                      | ✅       | ✅         | [YesianRohn](https://github.com/YesianRohn) |
| [BUSNet](./configs/rec/busnet/)               | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28402)                            | ✅       | ✅         |                                             |
| DCTC                                          | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28575)                            |          |            | TODO                                        |
| [CAM](./configs/rec/cam/)                     | [PR 2024](https://arxiv.org/abs/2402.13643)                                                    | ✅       | ✅         |                                             |
| [OTE](./configs/rec/ote/)                     | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.html) | ✅       | ✅         |                                             |
| CFF                                           | [IJCAI 2024](https://arxiv.org/abs/2407.05562)                                                 |          |            | TODO                                        |
| [DPTR](./configs/rec/dptr/)                   | [ACM MM 2024](https://dl.acm.org/doi/10.1145/3664647.3681390)                                  |          |            | [fd-zs](https://github.com/fd-zs)           |
| VIPTR                                         | [ACM CIKM 2024](https://arxiv.org/abs/2401.10110)                                              |          |            | TODO                                        |
| [IGTR](./configs/rec/igtr/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10820836)                                    | ✅       | ✅         |                                             |
| [SMTR](./configs/rec/smtr/)                   | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32285)                            | ✅       | ✅         |                                             |
| [CPPD](./configs/rec/cppd/)                   | [TPAMI 2025](https://ieeexplore.ieee.org/document/10902187)                                    | ✅       | ✅         |                                             |
| [FocalSVTR-CTC](./configs/rec/svtrs/)         | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32285)                            | ✅       | ✅         |                                             |
| [SVTRv2](./configs/rec/svtrv2/)               | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html) | ✅       | ✅         |                                             |
| [ResNet+Trans-CTC](./configs/rec/svtrs/)      |                                                                                                | ✅       | ✅         |                                             |
| [ViT-CTC](./configs/rec/svtrs/)               |                                                                                                | ✅       | ✅         |                                             |
| [MDiff4STR](./configs/rec/mdiff4str/)         | [AAAI 2025 Oral](https://arxiv.org/abs/2512.01422)                                             | ✅       | ✅         |                                             |

### Scene Text Detection (STD)

TODO

### Text Spotting

TODO

______________________________________________________________________

## Citation

If you find our method useful for your reserach, please cite:

```bibtex
@inproceedings{Du2025SVTRv2,
  title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
  author={Yongkun Du and Zhineng Chen and Hongtao Xie and Caiyan Jia and Yu-Gang Jiang},
  booktitle={ICCV},
  year={2025},
  pages={20147-20156}
}

@article{du2025unirec,
  title={UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters},
  author={Yongkun Du and Zhineng Chen and Yazhen Xie and Weikang Bai and Hao Feng and Wei Shi and Yuchen Su and Can Huang and Yu-Gang Jiang},
  journal={arXiv preprint arXiv:2512.21095},
  year={2025}
}
```

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!
