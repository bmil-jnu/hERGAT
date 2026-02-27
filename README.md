# hERGAT
Title: **hERGAT: Predicting hERG blockers using graph attention mechanism through atom- and molecule-level interaction analysis**  
Authors: *Dohyeon Lee, Sunyong Yoo*

---

## Folder structure

- `hergat/` : model + RDKit preprocessing + checkpoint + visualization utilities
- `scripts/` :
  - `train.py` : training + metrics export
  - `make_topk_figures.py` : TOP-5 figure export (molecule + atom level)
  - `run_all.py` : **one-command** pipeline (train → figures → summary)
- `data/` : hERGAT_final_dataset (CSV)

---

## Description

We present hERGAT, an interpretable deep learning model that utilizes graph attention to predict hERG blockers.

In addition, we provide a Python file that can be used to generate predictions from the model we trained.

If you want, you can train and predict new datasets from the structure of the model we proposed.

- [Data](https://github.com/DOHYEON7222/hERGAT/tree/main/data)
  
### 데이터 & 가중치 파일
현재 용량 문제로 인하여 github에는 올릴 수 없으니 google drive에 들어가셔서 직접 다운로드 받으셔야합니다.
데이터 파일은 data폴더에 위치해야하며, 모델 가중치 파일은 ckpt라는 폴더안에 위치해야합니다

| 파일명 | 다운로드 링크 | 저장 위치 |
|--------|--------------|-----------|
| `hERGAT_best.pt` | [Google Drive 링크](https://drive.google.com/file/d/1-t4sCvtmwkIuyVUBGS0XRw7IB3HeiP7I/view?usp=drive_link) | `outputs/paper_run1/ckpt/hERGAT_best.pt` |
| `hERGAT_final_dataset.csv` | [Google Drive 링크](https://drive.google.com/file/d/1kw66fRQmYfPPP_yMlEF_xvaggCt8Hq3G/view?usp=sharing) | `data/hERGAT_final_dataset.csv` |

---
- [hERG source code](https://github.com/DOHYEON7222/hERGAT/tree/main/hergat)
- [Atom- and molecule-level interaction analysis](https://github.com/DOHYEON7222/hERGAT/tree/main/outputs/paper_run1/figures)


## One-command run (train + evaluation + TOP-5 figures)

From the project root:

### terminal 
```bat
python scripts\run_all.py ^
  --data_csv data\hERGAT_final_dataset.csv ^
  --out_dir outputs\paper_run1 ^
  --smiles_col SMILES ^
  --label_col Class ^
  --device cuda:0 ^
  --top_k 5

```
Outputs (example):
- `outputs/paper_run1/ckpt/hERGAT_best.pt`
- `outputs/paper_run1/metrics.json`
- `outputs/paper_run1/roc_curve.json`, `outputs/paper_run1/pr_curve.json`
- `outputs/paper_run1/figures/topk_predictions.csv`
- `outputs/paper_run1/figures/topk_figures_summary.json`
- `outputs/paper_run1/paper_run_summary.json`

---

## Dependency

`Python == 3.9.18`
`tensorflow == 2.15.0`
`keras == 2.15.0`
`scikit-learn==1.3.2`
`rdkit==2023.9.2`


# Contacts

If you have any questions or comments, please feel free to create an issue on github here, or email us:

- ldhyun7222@naver.com
- syyoo@jnu.ac.kr
