def log_disabled() -> None:
    print("[Evaluation] Disabled by config: evaluation.enabled=false")


def log_module_eval(module_name: str, pred_dir: str) -> None:
    print(f"[Evaluation] Evaluating {module_name} from {pred_dir}")


def log_module_skip(module_name: str, pred_dir: str) -> None:
    print(f"[Evaluation] Skip {module_name}: missing {pred_dir}")


def log_done(output_dir: str) -> None:
    print(f"[Evaluation] Done. Output: {output_dir}")
