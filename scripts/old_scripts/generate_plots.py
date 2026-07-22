import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

DEV_FILE = "db/bird-1/dev.json"
OUTPUT_DIR = "data/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    {"name": "1. Baseline", "path": "logs/baseline_toxicology.json"},
    {"name": "2. Vector RAG", "path": "logs/rag_toxicology.json"},
    {"name": "3. Graph RAG", "path": "logs/graph_toxicology.json"},
    {"name": "4. Hybrid V1", "path": "logs/hybrid_toxicology.json"},
    {"name": "5. Hybrid V2", "path": "logs/hybrid_v2_toxicology.json"}
]

def load_benchmark_metadata(filepath):
    """Carrega el dev.json per extreure la dificultat de cada pregunta."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {item['question_id']: item.get('difficulty', 'unknown') 
            for item in data if item.get('db_id') == 'toxicology'}

def load_experiment_data(exp_dict, difficulty_map):
    """Carrega el JSON d'un experiment i el converteix en un DataFrame."""
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
        print(f"Error carregant {exp_dict['name']}: {e}")
        return pd.DataFrame()

print("Carregant metadades del benchmark...")
difficulty_map = load_benchmark_metadata(DEV_FILE)

print("Processant experiments...")
df_list = []
for exp in EXPERIMENTS:
    df_list.append(load_experiment_data(exp, difficulty_map))

df_all = pd.concat(df_list, ignore_index=True)


# Rendiment Global (Accuracy General)
plt.figure(figsize=(10, 6))
agg_df = df_all.groupby('architecture')['accuracy'].mean().reset_index()
agg_df['accuracy_pct'] = agg_df['accuracy'] * 100

ax = sns.barplot(data=agg_df, x='architecture', y='accuracy_pct', palette='viridis')
plt.title('Precisió Global per Arquitectura (Estudi d\'Ablació)', fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.xlabel('Arquitectura')
plt.ylim(0, 100)

# Afegir els percentatges a sobre de cada barra
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), 
                textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/overall_accuracy.png", dpi=300)
plt.close()

# Rendiment per Nivell de Dificultat
plt.figure(figsize=(12, 7))
# Ordenem les dificultats perquè tinguin sentit lògic
difficulty_order = ['Simple', 'Moderate', 'Challenging'] 
df_all['difficulty'] = pd.Categorical(df_all['difficulty'], categories=difficulty_order, ordered=True)

ax = sns.barplot(data=df_all, x='architecture', y='accuracy', hue='difficulty', palette='mako', errorbar=None)
plt.title('Comparativa de Precisió segons la Dificultat de la Consulta SQL', fontweight='bold')
plt.ylabel('Accuracy (0.0 - 1.0)')
plt.xlabel('Arquitectura')
plt.legend(title='Dificultat')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_by_difficulty.png", dpi=300)
plt.close()

# Taxa d'Error Relativa (Quant millora respecte al Baseline)
baseline_acc = agg_df[agg_df['architecture'] == '1. Baseline']['accuracy_pct'].values
if len(baseline_acc) > 0:
    agg_df['improvement'] = agg_df['accuracy_pct'] - baseline_acc[0]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agg_df[agg_df['architecture'] != '1. Baseline'], 
                     x='architecture', y='improvement', palette='crest')
    plt.title('Increment absolut de precisió respecte al Baseline', fontweight='bold')
    plt.ylabel('Millora (Punts Percentuals)')
    plt.xlabel('Arquitectura')
    
    for p in ax.patches:
        ax.annotate(f"+{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 8), 
                    textcoords='offset points', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/baseline_improvement.png", dpi=300)
    plt.close()

print(f"Gràfics generats amb èxit a la carpeta '{OUTPUT_DIR}'.")