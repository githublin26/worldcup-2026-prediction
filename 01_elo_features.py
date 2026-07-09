"""
Step 1: 資料清理 + Elo 評分系統
------------------------------------------
輸入：Kaggle 的 international-football-results-from-1872-to-2017（或更新版本）
      CSV 欄位通常長這樣：
      date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

輸出：帶有 Elo 特徵的乾淨資料表，存成 results_with_elo.csv
"""

import pandas as pd
import numpy as np

# ============ 參數設定（Kaggle Notebook 路徑版）============
# 注意：Kaggle 資料集加入後，實際資料夾名稱可能因版本略有不同，
# 若路徑對不上，Notebook 右側 "Add Input" 欄位下方會顯示正確路徑，複製貼上即可
INPUT_CSV = "/kaggle/input/datasets/martj42/international-football-results-from-1872-to-2017/results.csv"
OUTPUT_CSV = "/kaggle/working/results_with_elo.csv"
INITIAL_ELO = 1500
BASE_K = 30                        # 一般賽事的 K 值
WORLD_CUP_K = 45                   # 世界盃/重要賽事 K 值放大
HOME_ADVANTAGE = 60                # 主場優勢，加在主隊 Elo 上做預期勝率計算

# ============ 讀取與清理 ============
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 確保球隊名稱一致（去除多餘空白）
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    return df


def get_k_factor(tournament: str) -> int:
    """世界盃正賽/資格賽給更高權重"""
    if isinstance(tournament, str) and "World Cup" in tournament:
        return WORLD_CUP_K
    return BASE_K


def expected_score(elo_a: float, elo_b: float) -> float:
    """標準 Elo 預期勝率公式"""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


# ============ 逐場更新 Elo ============
def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    elo_dict = {}  # team -> current elo

    home_elo_before, away_elo_before = [], []

    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        elo_dict.setdefault(home, INITIAL_ELO)
        elo_dict.setdefault(away, INITIAL_ELO)

        elo_h, elo_a = elo_dict[home], elo_dict[away]
        home_elo_before.append(elo_h)
        away_elo_before.append(elo_a)

        # 計算實際比賽結果分數：贏=1, 平=0.5, 輸=0
        if row["home_score"] > row["away_score"]:
            score_h, score_a = 1, 0
        elif row["home_score"] < row["away_score"]:
            score_h, score_a = 0, 1
        else:
            score_h, score_a = 0.5, 0.5

        # 中立場地不加主場優勢
        adj_home = 0 if row.get("neutral", False) else HOME_ADVANTAGE
        exp_h = expected_score(elo_h + adj_home, elo_a)
        exp_a = 1 - exp_h

        k = get_k_factor(row.get("tournament", ""))
        elo_dict[home] = elo_h + k * (score_h - exp_h)
        elo_dict[away] = elo_a + k * (score_a - exp_a)

    df["home_elo"] = home_elo_before
    df["away_elo"] = away_elo_before
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df, elo_dict


# ============ 近期戰績特徵 ============
def add_recent_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """計算每隊在該場比賽之前，近 n 場的平均進球、失球、勝率"""
    team_history = {}  # team -> list of (goals_for, goals_against, result)

    home_avg_gf, home_avg_ga, home_form = [], [], []
    away_avg_gf, away_avg_ga, away_form = [], [], []

    for idx, row in df.iterrows():
        for side, team, gf_col, ga_col, avg_gf_list, avg_ga_list, form_list in [
            ("home", row["home_team"], "home_score", "away_score",
             home_avg_gf, home_avg_ga, home_form),
            ("away", row["away_team"], "away_score", "home_score",
             away_avg_gf, away_avg_ga, away_form),
        ]:
            hist = team_history.get(team, [])
            recent = hist[-n_matches:] if len(hist) > 0 else []
            if recent:
                avg_gf_list.append(np.mean([m[0] for m in recent]))
                avg_ga_list.append(np.mean([m[1] for m in recent]))
                form_list.append(np.mean([m[2] for m in recent]))
            else:
                avg_gf_list.append(np.nan)
                avg_ga_list.append(np.nan)
                form_list.append(np.nan)

        # 比賽結束後更新歷史紀錄
        h_result = 1 if row["home_score"] > row["away_score"] else (
            0.5 if row["home_score"] == row["away_score"] else 0)
        a_result = 1 - h_result if h_result != 0.5 else 0.5

        team_history.setdefault(row["home_team"], []).append(
            (row["home_score"], row["away_score"], h_result))
        team_history.setdefault(row["away_team"], []).append(
            (row["away_score"], row["home_score"], a_result))

    df["home_avg_gf"] = home_avg_gf
    df["home_avg_ga"] = home_avg_ga
    df["home_form"] = home_form
    df["away_avg_gf"] = away_avg_gf
    df["away_avg_ga"] = away_avg_ga
    df["away_form"] = away_form
    return df


# ============ 建立分類標籤 ============
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    choices = ["home_win", "draw", "away_win"]
    df["result"] = np.select(conditions, choices)
    return df


if __name__ == "__main__":
    df = load_data(INPUT_CSV)
    df, final_elo = compute_elo(df)
    df = add_recent_form(df, n_matches=5)
    df = add_label(df)

    # 丟掉早期資料不足的比賽（避免 near-cold-start 雜訊）
    df = df.dropna(subset=["home_avg_gf", "away_avg_gf"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"完成！輸出至 {OUTPUT_CSV}，共 {len(df)} 筆比賽資料")

    # 順便存最新的 Elo 排行榜，之後模擬 2026 世界盃會用到
    elo_ranking = pd.Series(final_elo).sort_values(ascending=False)
    elo_ranking.to_csv("/kaggle/working/latest_elo_ranking.csv", header=["elo"])
    print("最新 Elo 排行榜已存至 /kaggle/working/latest_elo_ranking.csv")
