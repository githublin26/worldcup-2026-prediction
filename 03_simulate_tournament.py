"""
Step 3: 蒙地卡羅模擬 2026 世界盃
------------------------------------------
用訓練好的泊松模型算出任兩隊對戰的比分機率分佈，
再模擬整個賽程（小組賽 → 16強 → 8強 → 4強 → 決賽）N 次，
統計每隊晉級/奪冠機率。

注意：groups.csv 需要你自己準備，欄位如下：
group,team,elo
A,USA,1750
A,Mexico,1700
...
knockout 對戰規則需等 2026 正式抽籤結果出爐後填入 KNOCKOUT_MAP
"""

import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from collections import defaultdict

N_SIMULATIONS = 20000
MAX_GOALS = 8  # 計算比分機率矩陣時的上限

# Kaggle Notebook 路徑版
poisson_model = joblib.load("/kaggle/working/model_poisson.pkl")
elo_ranking = pd.read_csv("/kaggle/working/latest_elo_ranking.csv", index_col=0)["elo"].to_dict()


def expected_goals(team_elo, opp_elo, team_avg_gf, opp_avg_ga, is_home):
    X = pd.DataFrame([{
        "team_elo": team_elo, "opp_elo": opp_elo,
        "team_avg_gf": team_avg_gf, "opp_avg_ga": opp_avg_ga,
        "is_home": is_home,
    }])
    return poisson_model.predict(X)[0]


def match_outcome_probs(team_a, team_b, avg_gf_map, avg_ga_map, neutral=True):
    """回傳 team_a 贏/平/輸的機率，以及比分機率矩陣（用於模擬進球數）"""
    lam_a = expected_goals(
        elo_ranking.get(team_a, 1500), elo_ranking.get(team_b, 1500),
        avg_gf_map.get(team_a, 1.2), avg_ga_map.get(team_b, 1.2),
        is_home=0 if neutral else 1,
    )
    lam_b = expected_goals(
        elo_ranking.get(team_b, 1500), elo_ranking.get(team_a, 1500),
        avg_gf_map.get(team_b, 1.2), avg_ga_map.get(team_a, 1.2),
        is_home=0,
    )

    score_matrix = np.outer(
        poisson.pmf(np.arange(MAX_GOALS + 1), lam_a),
        poisson.pmf(np.arange(MAX_GOALS + 1), lam_b),
    )
    p_a_win = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_b_win = np.triu(score_matrix, 1).sum()

    return p_a_win, p_draw, p_b_win, lam_a, lam_b


def simulate_group_stage(groups: pd.DataFrame, avg_gf_map, avg_ga_map):
    """回傳每組前兩名晉級隊伍"""
    standings = defaultdict(lambda: defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0}))

    for group_name, teams in groups.groupby("group")["team"]:
        teams = list(teams)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                p_a, p_draw, p_b, lam_a, lam_b = match_outcome_probs(
                    a, b, avg_gf_map, avg_ga_map)
                outcome = np.random.choice(["a", "draw", "b"], p=[p_a, p_draw, p_b])

                goals_a = np.random.poisson(lam_a)
                goals_b = np.random.poisson(lam_b)

                if outcome == "a":
                    standings[group_name][a]["pts"] += 3
                elif outcome == "b":
                    standings[group_name][b]["pts"] += 3
                else:
                    standings[group_name][a]["pts"] += 1
                    standings[group_name][b]["pts"] += 1

                standings[group_name][a]["gf"] += goals_a
                standings[group_name][a]["ga"] += goals_b
                standings[group_name][b]["gf"] += goals_b
                standings[group_name][b]["ga"] += goals_a

    qualified = {}
    for group_name, teams_dict in standings.items():
        ranked = sorted(
            teams_dict.items(),
            key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"]),
            reverse=True,
        )
        qualified[group_name] = [ranked[0][0], ranked[1][0]]
    return qualified


def simulate_knockout(team_a, team_b, avg_gf_map, avg_ga_map):
    """淘汰賽不能平局，平局時用純機率重新分配（簡化處理，未模擬PK）"""
    p_a, p_draw, p_b, _, _ = match_outcome_probs(team_a, team_b, avg_gf_map, avg_ga_map)
    p_a_adj = p_a + p_draw * (p_a / (p_a + p_b))
    return np.random.choice([team_a, team_b], p=[p_a_adj, 1 - p_a_adj])


def run_full_simulation(groups_csv="/kaggle/input/groups-2026/groups.csv", avg_gf_map=None, avg_ga_map=None):
    # ↑ 這是你自己上傳 groups.csv 後的路徑，資料夾名稱請依 Kaggle 實際顯示的路徑調整
    groups = pd.read_csv(groups_csv)
    avg_gf_map = avg_gf_map or {}
    avg_ga_map = avg_ga_map or {}

    champion_count = defaultdict(int)
    final4_count = defaultdict(int)

    for sim in range(N_SIMULATIONS):
        qualified = simulate_group_stage(groups, avg_gf_map, avg_ga_map)
        # 這裡的 16 強對戰表需依 2026 正式賽制設定（32隊擴軍版，B1 vs A2 之類規則）
        # 以下僅為示意，實際請替換成官方對戰規則
        round_of_16 = list(qualified.values())
        round_of_16_flat = [team for pair in round_of_16 for team in pair]

        current_round = round_of_16_flat
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                winner = simulate_knockout(
                    current_round[i], current_round[i + 1], avg_gf_map, avg_ga_map)
                next_round.append(winner)
            if len(next_round) == 4:
                for t in next_round:
                    final4_count[t] += 1
            current_round = next_round

        champion_count[current_round[0]] += 1

    results = pd.DataFrame({
        "team": list(champion_count.keys()),
        "champion_prob": [v / N_SIMULATIONS for v in champion_count.values()],
    }).sort_values("champion_prob", ascending=False)

    return results


if __name__ == "__main__":
    results = run_full_simulation()
    print(results.to_string(index=False))
    results.to_csv("/kaggle/working/champion_probabilities.csv", index=False)
