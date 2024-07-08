import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

from runner.run_manager import RunManager

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    pipeline_setup =    \
        {
            "keyword_extraction": {
                "engine": "gpt-3.5-turbo-0125",
                "temperature": 0.2,
                "base_uri": ""
            },
            "entity_retrieval": {
                "mode": "ask_model"
            },
            "context_retrieval": {
                "mode": "vector_db",
                "top_k": 5
            },
            "column_filtering": {
                "engine": "gpt-3.5-turbo-0125",
                "temperature": 0.0,
                "base_uri": ""
            },
            "table_selection": {
                "mode": "ask_model",
                "engine": "gpt-4-turbo",
                "temperature": 0.0,
                "base_uri": "",
                "sampling_count": 1
            },
            "column_selection": {
                "mode": "ask_model",
                "engine": "gpt-4-turbo",
                "temperature": 0.0,
                "base_uri": "",
                "sampling_count": 1
            },
            "candidate_generation": {
                "engine": "gpt-4-turbo",
                "temperature": 0.0,
                "base_uri": "",
                "sampling_count": 1
            },
            # "revision": {
            #     "engine": "gpt-4-turbo",
            #     "temperature": 0.0,
            #     "base_uri": "",
            #     "sampling_count": 1
            # }
        }
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, help="Mode of the data to be processed.", default="dev")
    parser.add_argument('--data_path', type=str, help="Path to the data file.", default="./data/dev/dev.json")
    parser.add_argument('--pipeline_nodes', type=str, help="Pipeline nodes configuration.", default="keyword_extraction+entity_retrieval+context_retrieval+column_filtering+table_selection+column_selection+candidate_generation")
    parser.add_argument('--pipeline_setup', type=str, help="Pipeline setup in JSON format.", default=json.dumps(pipeline_setup))
    parser.add_argument('--use_checkpoint', action='store_true', help="Flag to use checkpointing.", default=False)
    parser.add_argument('--checkpoint_nodes', type=str, required=False, help="Checkpoint nodes configuration.")
    parser.add_argument('--checkpoint_dir', type=str, required=False, help="Directory for checkpoints.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    if args.use_checkpoint:
        print('Using checkpoint')
        if not args.checkpoint_nodes:
            raise ValueError('Please provide the checkpoint nodes to use checkpoint')
        if not args.checkpoint_dir:
            raise ValueError('Please provide the checkpoint path to use checkpoint')

    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

if __name__ == '__main__':
    main()
