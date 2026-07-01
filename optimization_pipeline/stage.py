from optimization_pipeline.executor import execute_optimization

def run_optimization(config):
    if not config.get("optimization", {}).get("enabled", True):
        print("[Optimization] Disabled in config.")
        return
    
    print("[Optimization] Starting optimization pipeline...")
    execute_optimization(config)
    print("[Optimization] Finished optimization pipeline.")
