# Training Run
#

from data import run_setup, get_query_data, inspect_query_df
from pathlib import Path

if __name__ == '__main__':

    # Setup Run
    local_project_folder = Path.cwd().joinpath('NLP_Project')
    data_folder, run_results_folder = run_setup(local_project_folder, 'Py_Test_1')

    # Get source medical data and inspect it
    query_raw_df = get_query_data(data_folder, run_results_folder)
    inspect_query_df(query_raw_df)
