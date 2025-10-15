# Data Load & Embedding Functions
#

from datetime import datetime
from pathlib import Path

import pandas as pd
import uuid
from IPython.display import display
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Pipeline Run Setup
#
def run_setup(local_project_folder, run_name=''):

    # Folder paths
    data_folder = local_project_folder.joinpath('data')
    if not data_folder.exists():
        raise FileNotFoundError(f'{data_folder} does not exist')
    results_folder = local_project_folder.joinpath('run_results')
    if not results_folder.exists():
        raise FileNotFoundError(f'{results_folder} does not exist')

    # Run Results
    run_name = f'{run_name.strip()}'
    run_name = f'Run_{run_name}' if (run_name) else f'Run_{datetime.now().strftime("%Y%m%d")}'
    run_results_folder = results_folder.joinpath(f'{run_name}')
    run_results_folder.mkdir(parents=True, exist_ok=True) 

    return data_folder, run_results_folder

# Load Source Datsets, Combine and Create df
#
def get_query_data(data_folder, run_results_folder):
    """
    Combines two datsets & saves it
    - medicheck-expert.csv using expert classifications 0: as 'Non-Medical' (1) and 1,2,3 as 'Medical' (0)
    - medicheck-neg.csv as 'Non-Medical'

    Args:
        None

    Returns:
        questions.df: df with sentenaces and classification label
    """

    # Load two datasets & combine
    # NB query_label_expert = 999 denotes no expert labeling
    expert_df = pd.read_csv(data_folder.joinpath('medicheck-expert.csv'))
    neg_df = pd.read_csv(data_folder.joinpath('medicheck-neg.csv'), header=None, names = ['query', 'query-label-expert'], escapechar='\\')
    neg_df['query-label-expert'] = 999

    # Combine the two datasets
    expert_df = expert_df[['query', 'query-label-expert']]
    combined_df = pd.concat([expert_df, neg_df], ignore_index=True)

    # Classify the querys into types
    # 'Non-Medical' (1) and 'Medical' (0)
    combined_df['query-is-non-medical'] = combined_df['query-label-expert'].isin([0,999])

    # Add a unique ref-id for later back-tracking
    combined_df.insert(0, 'ref-id', [str(uuid.uuid4())[:8] for _ in range(len(combined_df))])

    # Save to raw dataset
    combined_df.to_pickle(run_results_folder.joinpath('query_data_raw.pkl'))

    return combined_df

# Inspect a query df
#

def inspect_query_df(query_df):
    print(f"Full Dataset shape: {query_df.shape}")
    display(query_df.head(10))

    # Plot unbalanced labels
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    query_df['query-label-expert'].value_counts().sort_index().plot(kind='bar', ax=axs[0])
    axs[0].set_title('query-label-expert')
    axs[0].set_xlabel('0=Non-medical, 1=Non-serious, 2=Serious, 3=Critical, 999=Non-Expert')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=0)

    query_df['query-is-non-medical'].value_counts().plot(kind='bar', ax=axs[1], color=['red', 'green'])
    axs[1].set_title('query-is-non-medical')
    axs[1].set_xlabel('True: 0,99, False: 1,2,3')
    axs[1].set_ylabel('Count')
    axs[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()
