import hashlib
import logging
import os


SUBJECTS = ["A", "B", "C", "D", "E", "F"]


def setup_logging(log_file_path, verbose):
    logger = logging.getLogger("search_pipeline")
    logger.setLevel(logging.INFO if verbose > 0 else logging.WARNING)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_thread_env(inner_threads):
    value = str(int(inner_threads))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["BLIS_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def run_tag(args):
    payload = repr(
        (
            args.cv_folds,
            args.start_min,
            args.start_max,
            args.start_step,
            args.stop_min,
            args.stop_max,
            args.stop_step,
            args.min_window_len,
            args.top_per_model,
            args.max_candidates_per_subject,
            args.min_ensemble_size,
            args.max_ensemble_size,
            args.max_same_model,
            args.max_same_family,
            args.min_window_shift,
            args.min_disagreement,
            args.diversity_weight,
            args.double_fault_weight,
            args.sample_rate_hz,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def make_window_grid(n_times, args):
    starts = list(range(args.start_min, args.start_max + 1, args.start_step))
    stops = list(range(args.stop_min, args.stop_max + 1, args.stop_step))
    if n_times not in stops:
        stops.append(n_times)

    windows = []
    for start in starts:
        for stop in stops:
            if stop > n_times:
                continue
            if stop - start < args.min_window_len:
                continue
            if stop <= start:
                continue
            windows.append((start, stop))

    windows.append((0, n_times))
    return sorted(set(windows))


def estimate_fit_counts(subjects, model_specs, windows, cv_folds):
    candidates = len(subjects) * len(model_specs) * len(windows)
    return candidates, candidates * cv_folds
