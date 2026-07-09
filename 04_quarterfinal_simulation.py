import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

N_SIMULATIONS = 20000
MAX_GOALS = 8

poisson_model = joblib.load("/kaggle/working/model_poisson.pkl")
elo_ranking = pd.read_csv("/kaggle/working/latest_elo_ranking.csv", index_col=0)["elo"].to_dict()

QUARTERFINALS = [
    ("Morocco", "France"),
    ("England", "Norway"),
    ("Spain", "Belgium"),
    ("Argentina", "Switzerland"),
]

TEAMS = [t for pair in QUARTERFINALS for t in pair]

def get_elo(team):
    for key in elo_ranking:
        if key.lower() == team.lower():
            return elo_ranking[key]
    return 1600

# ---- 關鍵優化：一次算好所有球隊的期望進球數，不在迴圈中重複呼叫 predict() ----
elo_map = {t: get_elo(t) for t in TEAMS}

# 建一個批次表，一次算完所有兩兩配對的期望進球（8隊只有 8*7=56 種排列組合，量很小）
batch_rows = []
pairs_order = []
for a in TEAMS:
    for b in TEAMS:
        if a == b:
            continue
        batch_rows.append({
            "team_elo": elo_map[a], "opp_elo": elo_map[b],
            "team_avg_gf": 1.5, "opp_avg_ga": 1.2, "is_home": 0,
        })
        pairs_order.append((a, b))

batch_df = pd.DataFrame(batch_rows)
batch_pred = poisson_model.predict(batch_df)  # 一次性預測，不是逐筆呼叫

lam_lookup = {pair: pred for pair, pred in zip(pairs_order, batch_pred)}

# ---- 預先算好所有可能對戰的勝率，模擬時直接查表 ----
win_prob_cache = {}
for a in TEAMS:
    for b in TEAMS:
        if a == b:
            continue
        lam_a = lam_lookup[(a, b)]
        lam_b = lam_lookup[(b, a)]
        matrix = np.outer(
            poisson.pmf(np.arange(MAX_GOALS + 1), lam_a),
            poisson.pmf(np.arange(MAX_GOALS + 1), lam_b),
        )
        p_a = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_b = np.triu(matrix, 1).sum()
        p_a_adj = p_a + p_draw * (p_a / (p_a + p_b))
        win_prob_cache[(a, b)] = p_a_adj

print("=== 8強單場勝率預估 ===")
for a, b in QUARTERFINALS:
    p = win_prob_cache[(a, b)]
    print(f"{a} 勝率 {p:.1%}  vs  {b} 勝率 {1-p:.1%}")

def simulate_knockout(team_a, team_b):
    p_a = win_prob_cache[(team_a, team_b)]
    return np.random.choice([team_a, team_b], p=[p_a, 1 - p_a])

champion_count = {}
final_count = {}
semifinal_count = {}

for _ in range(N_SIMULATIONS):
    semifinalists = [simulate_knockout(a, b) for a, b in QUARTERFINALS]
    for t in semifinalists:
        semifinal_count[t] = semifinal_count.get(t, 0) + 1
    finalists = [
        simulate_knockout(semifinalists[0], semifinalists[1]),
        simulate_knockout(semifinalists[2], semifinalists[3]),
    ]
    for t in finalists:
        final_count[t] = final_count.get(t, 0) + 1
    champion = simulate_knockout(finalists[0], finalists[1])
    champion_count[champion] = champion_count.get(champion, 0) + 1

results = pd.DataFrame({
    "team": TEAMS,
    "champion_prob": [champion_count.get(t, 0) / N_SIMULATIONS for t in TEAMS],
    "final_prob": [final_count.get(t, 0) / N_SIMULATIONS for t in TEAMS],
    "semifinal_prob": [semifinal_count.get(t, 0) / N_SIMULATIONS for t in TEAMS],
}).sort_values("champion_prob", ascending=False)

print("\n=== 2026世界盃奪冠機率模擬（基於8強現況）===")
print(results.to_string(index=False))

results.to_csv("/kaggle/working/champion_probabilities_qf.csv", index=False)
print("\n結果已存至 /kaggle/working/champion_probabilities_qf.csv")
