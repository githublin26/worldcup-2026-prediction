# 🏆 2026 世界盃冠軍預測模型

用 Elo 評分系統、LightGBM 分類模型與泊松迴歸，預測 2026 FIFA 世界盃各隊奪冠機率。

## 專案簡介

這個專案基於 1872–2017 年的國際足球比賽歷史資料，建立了三層預測架構：

1. **Elo 評分系統**：追蹤每支國家隊的實力變化，世界盃正賽對戰給予更高權重
2. **LightGBM 三分類模型**：預測單場比賽勝／平／負
3. **泊松迴歸模型**：預測雙方預期進球數，用於模擬精確比分與淘汰賽晉級

最終透過蒙地卡羅模擬（Monte Carlo Simulation）跑 20,000 次淘汰賽路徑，統計出各隊奪冠機率。

## 📊 目前結果（截至 2026 世界盃 8 強階段）

模擬時間：2026 世界盃 8 強確定後，蒙地卡羅模擬 20,000 次

| 排名 | 球隊 | 奪冠機率 | 晉級決賽機率 | 晉級4強機率 |
|------|------|----------|--------------|--------------|
| 🥇 1 | 阿根廷 | 24.6% | 40.2% | 70.9% |
| 🥈 2 | 西班牙 | 21.7% | 36.3% | 66.7% |
| 3 | 法國 | 18.7% | 37.3% | 61.6% |
| 4 | 英格蘭 | 12.0% | 27.7% | 58.3% |
| 5 | 摩洛哥 | 7.4% | 18.5% | 38.4% |
| 6 | 挪威 | 5.8% | 16.4% | 41.7% |
| 7 | 比利時 | 5.4% | 12.8% | 33.5% |
| 8 | 瑞士 | 4.3% | 10.7% | 29.1% |

> 完整結果請見 [`results/champion_probabilities_qf.csv`](results/champion_probabilities_qf.csv)

**8強單場勝率預估：**
- 摩洛哥 38.4% vs 法國 61.6%
- 英格蘭 58.3% vs 挪威 41.7%
- 西班牙 66.5% vs 比利時 33.5%
- 阿根廷 70.3% vs 瑞士 29.7%

## 🛠 技術棧

- **語言**：Python 3.12
- **建模**：LightGBM, statsmodels (GLM/Poisson)
- **資料處理**：pandas, numpy
- **評估**：scikit-learn
- **執行環境**：Kaggle Notebook

## 📁 專案結構

```
worldcup-2026-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── worldcup_prediction.ipynb    # 完整 Kaggle Notebook（含執行結果）
├── src/
│   ├── 01_elo_features.py           # Elo 評分計算 + 特徵工程
│   ├── 02_train_models.py           # LightGBM + 泊松迴歸訓練
│   ├── 03_simulate_tournament.py    # 蒙地卡羅賽事模擬（完整賽程版）
│   └── 04_quarterfinal_simulation.py # 蒙地卡羅模擬（8強現況版，效能優化）
├── results/
│   └── champion_probabilities_qf.csv
└── .gitignore
```

## 🚀 如何執行

### 1. 安裝套件
```bash
pip install -r requirements.txt
```

### 2. 下載資料集
本專案使用 Kaggle 資料集，請自行下載並放入 `data/` 資料夾：
[International football results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

### 3. 依序執行
```bash
python src/01_elo_features.py
python src/02_train_models.py
python src/03_simulate_tournament.py
```

## 📈 方法論

### Elo 評分系統
- 初始分數 1500，依賽事重要性調整 K 值（世界盃正賽 K=45，一般賽事 K=30）
- 加入主場優勢修正（+60 分）

### 模型評估
- 訓練/測試依時間序列切分，避免用未來資料預測過去
- LightGBM 三分類準確率約 60%（隨機猜測基準線為 33%）
- 已知限制：平局預測召回率偏低，這是足球預測模型的常見挑戰

### 淘汰賽模擬
- 用泊松分布計算雙方比分機率矩陣
- 平局時依相對勝率重新分配（未模擬延長賽/PK 細節）

## ⚠️ 已知限制

- 訓練資料截至 2017 年，未涵蓋近期陣容、傷病等即時資訊
- 平局預測能力較弱，可透過 Dixon-Coles 修正改善
- 淘汰賽 PK 大戰未單獨建模，簡化為相對機率分配

## 授權

僅供學習與研究用途，資料集版權歸原始 Kaggle 資料集作者所有。
