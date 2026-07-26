import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate accuracy plots for a specific database.")
parser.add_argument("--db", type=str, required=True, help="Target database name (e.g., toxicology, financial)")
args = parser.parse_args()

DB_NAME = args.db

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

DEV_FILE = "db/bird-1/dev.json"
OUTPUT_DIR = f"data/plots/{DB_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standardized paths
EXPERIMENTS = [
    {"name": "1. Baseline", "path": f"logs/{DB_NAME}/baseline_{DB_NAME}.json"},
    {"name": "2. Vector RAG", "path": f"logs/{DB_NAME}/vector_{DB_NAME}.json"},
    {"name": "3. Graph RAG", "path": f"logs/{DB_NAME}/graph_{DB_NAME}.json"},
    {"name": "4. Hybrid V1", "path": f"logs/{DB_NAME}/hybrid_v1_{DB_NAME}.json"},
    {"name": "5. Hybrid V2", "path": f"logs/{DB_NAME}/hybrid_v2_{DB_NAME}.json"}
]

def load_benchmark_metadata(filepath, db_target):
    """Loads dev.json to extract the difficulty of each question."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {item['question_id']: item.get('difficulty', 'unknown') 
            for item in data if item.get('db_id') == db_target}

def load_experiment_data(exp_dict, difficulty_map):
    """Loads an experiment JSON and converts it into a DataFrame."""
    try:
        with open(exp_dict['path'], 'r') as f:
            data = json.load(f)
        
        queries = data.get('detailed_queries', data)
        
        records = []
        for q in queries:
            q_id = q.get('query_id')
            records.append({
                'query_id': q_id,
                'architecture': exp_dict['name'],
                'accuracy': float(q.get('accuracy', 0.0)),
                'difficulty': difficulty_map.get(q_id, 'unknown').capitalize()
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error loading {exp_dict['name']}: {e}")
        return pd.DataFrame()

print(f"Loading benchmark metadata for DB: {DB_NAME}...")
difficulty_map = load_benchmark_metadata(DEV_FILE, DB_NAME)

print("Processing experiments...")
df_list = []
for exp in EXPERIMENTS:
    df_list.append(load_experiment_data(exp, difficulty_map))

df_all = pd.concat(df_list, ignore_index=True)


# Global Performance (Overall Accuracy)
plt.figure(figsize=(10, 6))
agg_df = df_all.groupby('architecture')['accuracy'].mean().reset_index()
agg_df['accuracy_pct'] = agg_df['accuracy'] * 100

ax = sns.barplot(data=agg_df, x='architecture', y='accuracy_pct', palette='viridis')
plt.title(f'Precisió Global per Arquitectura ({DB_NAME})', fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.xlabel('Arquitectura')
plt.ylim(0, 100)

# Add percentages on top of each bar
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), 
                textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{DB_NAME}_overall_accuracy.png", dpi=300)
plt.close()

# Performance by Difficulty Level
plt.figure(figsize=(12, 7))
# Order difficulties logically
difficulty_order = ['Simple', 'Moderate', 'Challenging'] 
df_all['difficulty'] = pd.Categorical(df_all['difficulty'], categories=difficulty_order, ordered=True)

ax = sns.barplot(data=df_all, x='architecture', y='accuracy', hue='difficulty', palette='mako', errorbar=None)
plt.title(f'Comparativa de precisió segons la dificultat de la consulta SQL ({DB_NAME})', fontweight='bold')
plt.ylabel('Accuracy (0.0 - 1.0)')
plt.xlabel('Arquitectura')
plt.legend(title='Dificultat')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{DB_NAME}_accuracy_by_difficulty.png", dpi=300)
plt.close()

# Relative Error Rate (Improvement over Baseline)
baseline_acc = agg_df[agg_df['architecture'] == '1. Baseline']['accuracy_pct'].values
if len(baseline_acc) > 0:
    agg_df['improvement'] = agg_df['accuracy_pct'] - baseline_acc[0]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agg_df[agg_df['architecture'] != '1. Baseline'], 
                     x='architecture', y='improvement', palette='crest')
    plt.title(f'Increment absolut de precisió respecte al Baseline ({DB_NAME})', fontweight='bold')
    plt.ylabel('Millora (Punts Percentuals)')
    plt.xlabel('Arquitectura')
    
    for p in ax.patches:
        ax.annotate(f"+{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 8), 
                    textcoords='offset points', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{DB_NAME}_baseline_improvement.png", dpi=300)
    plt.close()

print(f"Plots successfully generated in '{OUTPUT_DIR}' directory.")