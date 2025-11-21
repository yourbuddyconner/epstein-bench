import json
import os
import glob
from datetime import datetime

BENCHMARK_DATA_DIR = "docs/benchmark_data"
INDEX_FILE = os.path.join(BENCHMARK_DATA_DIR, "index.json")

def main():
    if not os.path.exists(BENCHMARK_DATA_DIR):
        os.makedirs(BENCHMARK_DATA_DIR)
        print(f"Created directory {BENCHMARK_DATA_DIR}")

    result_files = glob.glob(os.path.join(BENCHMARK_DATA_DIR, "*.json"))
    result_files = [f for f in result_files if not f.endswith("index.json")]
    
    summary_list = []
    
    for fpath in result_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract key metrics for the summary
            metrics = data.get("metrics", {})
            config = data.get("config", {})
            top_k = config.get("top_k", 5)
            
            # Try to find recall at top_k, or fallback to any recall
            recall_key = f"recall@{top_k}"
            if recall_key not in metrics:
                # Fallback: look for any key starting with recall@
                recalls = [k for k in metrics.keys() if k.startswith("recall@")]
                recall_key = recalls[0] if recalls else "recall@5"
            
            # Try to find latency
            latency_key = "mean_latency[end_to_end]"
            if latency_key not in metrics:
                 # Fallback: look for any key starting with mean_latency
                 latencies = [k for k in metrics.keys() if k.startswith("mean_latency")]
                 latency_key = latencies[0] if latencies else "mean_latency"

            # Try to find BERTScore (OpenAI or local)
            bertscore_key = "openai_bertscore_f1_reference"
            if bertscore_key not in metrics:
                bertscore_key = "bertscore_f1_reference"
                
            # Try to find LLM Answer Quality
            llm_quality_key = "llm_answer_quality"

            summary = {
                "filename": os.path.basename(fpath),
                "system_name": data.get("system_name", "Unknown"),
                "timestamp": data.get("timestamp", ""),
                "dataset": data.get("dataset", ""),
                "recall": metrics.get(recall_key, {}).get("value", "N/A"),
                "recall_k": recall_key,
                "token_f1": metrics.get("token_f1", {}).get("value", "N/A"),
                "bertscore": metrics.get(bertscore_key, {}).get("value", "N/A"),
                "llm_quality": metrics.get(llm_quality_key, {}).get("value", "N/A"),
                "latency": metrics.get(latency_key, {}).get("value", "N/A"),
            }
            summary_list.append(summary)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Sort by timestamp descending
    summary_list.sort(key=lambda x: x["timestamp"], reverse=True)
    
    with open(INDEX_FILE, "w") as f:
        json.dump(summary_list, f, indent=2)
    
    print(f"Updated {INDEX_FILE} with {len(summary_list)} entries.")

if __name__ == "__main__":
    main()

