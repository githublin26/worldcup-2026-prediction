"""
Step 2: 模型訓練
------------------------------------------
讀取 01_elo_features.py 產出的 results_with_elo.csv
訓練兩種模型：
  A) LightGBM 三分類（主勝/平/客勝）
  B) 泊松迴歸雙模型（分別預測主隊、客隊進球數）→ 之後可模擬任意比分機率
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib

# Kaggle Notebook 路徑版：讀取 Cell 2（01_elo_features）產出的檔案
INPUT_CSV = "/kaggle/working/results_with_elo.csv"

FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "home_avg_gf", "home_avg_ga", "home_form",
    "away_avg_gf", "away_avg_ga", "away_form",
]

# ============ 讀取資料 ============
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 依時間切分：最後 10% 當測試集（模擬「用過去預測未來」）
split_idx = int(len(df) * 0.9)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"訓練集: {len(train_df)} 筆 | 測試集: {len(test_df)} 筆")

# ============ A) LightGBM 三分類模型 ============
le = LabelEncoder()
y_train = le.fit_transform(train_df["result"])   # home_win / draw / away_win
y_test = le.transform(test_df["result"])

X_train = train_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

pred = clf.predict(X_test)
pred_proba = clf.predict_proba(X_test)

print("\n=== LightGBM 分類結果 ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Log Loss:", log_loss(y_test, pred_proba))
print(classification_report(y_test, pred, target_names=le.classes_))

joblib.dump(clf, "/kaggle/working/model_classifier.pkl")
joblib.dump(le, "/kaggle/working/label_encoder.pkl")

# ============ B) 泊松迴歸：分別預測主客進球數 ============
# 把「每場比賽」拆成「主隊視角」與「客隊視角」兩列，方便建立對稱模型
poisson_rows = []
for _, row in train_df.iterrows():
    poisson_rows.append({
        "goals": row["home_score"],
        "team_elo": row["home_elo"],
        "opp_elo": row["away_elo"],
        "team_avg_gf": row["home_avg_gf"],
        "opp_avg_ga": row["away_avg_ga"],
        "is_home": 1,
    })
    poisson_rows.append({
        "goals": row["away_score"],
        "team_elo": row["away_elo"],
        "opp_elo": row["home_elo"],
        "team_avg_gf": row["away_avg_gf"],
        "opp_avg_ga": row["home_avg_ga"],
        "is_home": 0,
    })

poisson_df = pd.DataFrame(poisson_rows)

poisson_model = smf.glm(
    formula="goals ~ team_elo + opp_elo + team_avg_gf + opp_avg_ga + is_home",
    data=poisson_df,
    family=sm.families.Poisson(),
).fit()

print("\n=== 泊松迴歸模型摘要 ===")
print(poisson_model.summary())

joblib.dump(poisson_model, "/kaggle/working/model_poisson.pkl")

print("\n模型已存檔：/kaggle/working/model_classifier.pkl, /kaggle/working/model_poisson.pkl")
