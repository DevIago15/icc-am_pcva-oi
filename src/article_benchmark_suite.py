from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from am_pcva_oi_base import random_euclidean_instance
from benchmark_grover_backends import (
    discover_approaches as discover_grover_approaches,
    run_solver as run_grover_solver,
)
from benchmark_policies import (
    discover_approaches as discover_policy_approaches,
    run_solver as run_policy_solver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "article_suite"


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("A lista de inteiros nao pode ser vazia.")
    return values


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    if df.empty:
        return "_sem dados_"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.{float_digits}f}")

    header = "| " + " | ".join(map(str, display_df.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display_df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def add_relative_delta_columns(
    df: pd.DataFrame,
    baseline_label: str,
    metric_to_delta_name: dict[str, str],
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_columns = list(group_columns or [])
    result = df.copy()
    for delta_name in metric_to_delta_name.values():
        result[delta_name] = math.nan

    if group_columns:
        group_key = group_columns[0] if len(group_columns) == 1 else group_columns
        grouped = result.groupby(group_key, dropna=False)
        for _, index in grouped.groups.items():
            group_df = result.loc[index]
            baseline_rows = group_df[group_df["approach"] == baseline_label]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            for metric_name, delta_name in metric_to_delta_name.items():
                baseline_value = float(baseline[metric_name])
                if abs(baseline_value) <= 1e-12:
                    continue
                result.loc[index, delta_name] = (
                    (result.loc[index, metric_name] - baseline_value) / baseline_value
                ) * 100.0
        return result

    baseline_rows = result[result["approach"] == baseline_label]
    if baseline_rows.empty:
        return result

    baseline = baseline_rows.iloc[0]
    for metric_name, delta_name in metric_to_delta_name.items():
        baseline_value = float(baseline[metric_name])
        if abs(baseline_value) <= 1e-12:
            continue
        result[delta_name] = ((result[metric_name] - baseline_value) / baseline_value) * 100.0
    return result


def compute_paired_comparison(
    df: pd.DataFrame,
    baseline_label: str,
    metric_name: str,
    runtime_metric_name: str = "runtime_seconds",
) -> pd.DataFrame:
    baseline = (
        df[df["approach"] == baseline_label][
            ["instance_size", "instance_seed", "solver_seed", metric_name, runtime_metric_name]
        ]
        .rename(
            columns={
                metric_name: "baseline_metric",
                runtime_metric_name: "baseline_runtime",
            }
        )
    )

    merged = df.merge(
        baseline,
        on=["instance_size", "instance_seed", "solver_seed"],
        how="left",
    )
    merged["metric_gap_vs_base"] = merged[metric_name] - merged["baseline_metric"]
    merged["runtime_gap_vs_base_seconds"] = merged[runtime_metric_name] - merged["baseline_runtime"]
    merged["win_vs_base"] = (merged["metric_gap_vs_base"] < -1e-12).astype(int)
    merged["loss_vs_base"] = (merged["metric_gap_vs_base"] > 1e-12).astype(int)
    merged["tie_vs_base"] = (
        (~merged["win_vs_base"].astype(bool)) & (~merged["loss_vs_base"].astype(bool))
    ).astype(int)

    return (
        merged.groupby("approach")
        .agg(
            paired_mean_cost_gap_vs_base=("metric_gap_vs_base", "mean"),
            paired_mean_runtime_gap_vs_base_seconds=("runtime_gap_vs_base_seconds", "mean"),
            paired_win_rate_vs_base=("win_vs_base", "mean"),
            paired_loss_rate_vs_base=("loss_vs_base", "mean"),
            paired_tie_rate_vs_base=("tie_vs_base", "mean"),
        )
        .reset_index()
    )


def aggregate_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("approach")
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            std_runtime_seconds=("runtime_seconds", "std"),
            mean_local_search_calls_total=("local_search_calls_total", "mean"),
            mean_local_search_calls_main_loop=("local_search_calls_main_loop", "mean"),
            mean_local_search_calls_initialization=("local_search_calls_initialization", "mean"),
            mean_local_search_skipped=("local_search_skipped", "mean"),
            mean_local_search_improvements=("local_search_improvements", "mean"),
            mean_local_search_activation_rate=("local_search_activation_rate", "mean"),
            mean_local_search_total_time_ms=("local_search_total_time_ms", "mean"),
            mean_local_search_total_delta_cost=("local_search_total_delta_cost", "mean"),
        )
        .reset_index()
        .sort_values(["mean_best_cost", "mean_runtime_seconds"], ascending=[True, True])
    )

    summary = add_relative_delta_columns(
        summary,
        baseline_label="am_pcva_oi_base",
        metric_to_delta_name={
            "mean_best_cost": "delta_best_cost_vs_base_pct",
            "mean_runtime_seconds": "delta_runtime_vs_base_pct",
            "mean_local_search_calls_total": "delta_ls_calls_vs_base_pct",
        },
    )

    paired = compute_paired_comparison(df, baseline_label="am_pcva_oi_base", metric_name="best_cost")
    return summary.merge(paired, on="approach", how="left")


def aggregate_policy_by_size(df: pd.DataFrame) -> pd.DataFrame:
    by_size = (
        df.groupby(["approach", "instance_size"])
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            mean_local_search_calls_total=("local_search_calls_total", "mean"),
            mean_local_search_calls_main_loop=("local_search_calls_main_loop", "mean"),
            mean_local_search_improvements=("local_search_improvements", "mean"),
            mean_local_search_activation_rate=("local_search_activation_rate", "mean"),
        )
        .reset_index()
        .sort_values(["instance_size", "mean_best_cost", "mean_runtime_seconds"], ascending=[True, True, True])
    )

    return add_relative_delta_columns(
        by_size,
        baseline_label="am_pcva_oi_base",
        metric_to_delta_name={
            "mean_best_cost": "delta_best_cost_vs_base_pct",
            "mean_runtime_seconds": "delta_runtime_vs_base_pct",
            "mean_local_search_calls_total": "delta_ls_calls_vs_base_pct",
        },
        group_columns=["instance_size"],
    )


def aggregate_grover_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("approach")
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            std_runtime_seconds=("runtime_seconds", "std"),
            mean_grover_calls=("grover_calls", "mean"),
            mean_grover_successes=("grover_successes", "mean"),
            mean_grover_candidate_pool=("grover_mean_candidate_pool_size", "mean"),
            mean_grover_backend_time_ms=("grover_total_backend_time_ms", "mean"),
            mean_grover_total_improvement=("grover_total_improvement", "mean"),
        )
        .reset_index()
        .sort_values(["mean_best_cost", "mean_runtime_seconds"], ascending=[True, True])
    )

    summary = add_relative_delta_columns(
        summary,
        baseline_label="am_pcva_oi_base_2opt",
        metric_to_delta_name={
            "mean_best_cost": "delta_best_cost_vs_base_pct",
            "mean_runtime_seconds": "delta_runtime_vs_base_pct",
        },
    )

    paired = compute_paired_comparison(df, baseline_label="am_pcva_oi_base_2opt", metric_name="best_cost")
    return summary.merge(paired, on="approach", how="left")


def aggregate_grover_by_size(df: pd.DataFrame) -> pd.DataFrame:
    by_size = (
        df.groupby(["approach", "instance_size"])
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            mean_grover_calls=("grover_calls", "mean"),
            mean_grover_successes=("grover_successes", "mean"),
            mean_grover_backend_time_ms=("grover_total_backend_time_ms", "mean"),
        )
        .reset_index()
        .sort_values(["instance_size", "mean_best_cost", "mean_runtime_seconds"], ascending=[True, True, True])
    )

    return add_relative_delta_columns(
        by_size,
        baseline_label="am_pcva_oi_base_2opt",
        metric_to_delta_name={
            "mean_best_cost": "delta_best_cost_vs_base_pct",
            "mean_runtime_seconds": "delta_runtime_vs_base_pct",
        },
        group_columns=["instance_size"],
    )


def build_grover_scaling_table(candidate_pool_sizes: Sequence[int], num_solutions: int) -> pd.DataFrame:
    rows = []
    for candidate_pool_size in candidate_pool_sizes:
        classical_queries = candidate_pool_size
        grover_iterations = max(
            1,
            math.floor((math.pi / 4.0) * math.sqrt(candidate_pool_size / max(num_solutions, 1))),
        )
        oracle_calls = 2 * grover_iterations + 1
        rows.append(
            {
                "candidate_pool_size": candidate_pool_size,
                "assumed_num_solutions": num_solutions,
                "classical_query_cost": classical_queries,
                "grover_iterations_estimate": grover_iterations,
                "grover_oracle_calls_estimate": oracle_calls,
                "ideal_query_speedup_vs_classical": classical_queries / oracle_calls,
            }
        )
    return pd.DataFrame(rows)


def run_policy_suite(
    instance_sizes: Sequence[int],
    instance_seeds: Sequence[int],
    solver_seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    approaches = discover_policy_approaches()
    results = []
    total_runs = len(instance_sizes) * len(instance_seeds) * len(solver_seeds) * len(approaches)
    current = 0

    print("=" * 80)
    print("[ARTICLE] Benchmark de policies")
    print(f"[INFO] Abordagens: {len(approaches)}")
    print(f"[INFO] Execucoes: {total_runs}")
    print("=" * 80)

    for n in instance_sizes:
        for instance_seed in instance_seeds:
            dist = random_euclidean_instance(n=n, seed=instance_seed)
            for solver_seed in solver_seeds:
                for approach in approaches:
                    current += 1
                    print(
                        f"[POLICIES {current}/{total_runs}] {approach['label']} | "
                        f"n={n} | inst_seed={instance_seed} | solver_seed={solver_seed}"
                    )
                    solver = approach["builder"](dist, solver_seed)
                    results.append(
                        run_policy_solver(
                            approach["label"],
                            solver,
                            n,
                            instance_seed,
                            solver_seed,
                        )
                    )

    detailed = pd.DataFrame(results)
    summary = aggregate_policy_summary(detailed)
    by_size = aggregate_policy_by_size(detailed)
    return detailed, summary, by_size


def run_grover_suite(
    instance_sizes: Sequence[int],
    instance_seeds: Sequence[int],
    solver_seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    approaches = discover_grover_approaches()
    results = []
    total_runs = len(instance_sizes) * len(instance_seeds) * len(solver_seeds) * len(approaches)
    current = 0

    print("=" * 80)
    print("[ARTICLE] Benchmark da trilha Grover")
    print(f"[INFO] Abordagens: {len(approaches)}")
    print(f"[INFO] Execucoes: {total_runs}")
    print("=" * 80)

    for n in instance_sizes:
        for instance_seed in instance_seeds:
            dist = random_euclidean_instance(n=n, seed=instance_seed)
            for solver_seed in solver_seeds:
                for approach in approaches:
                    current += 1
                    print(
                        f"[GROVER {current}/{total_runs}] {approach['label']} | "
                        f"n={n} | inst_seed={instance_seed} | solver_seed={solver_seed}"
                    )
                    solver = approach["builder"](dist, solver_seed)
                    results.append(
                        run_grover_solver(
                            approach["label"],
                            solver,
                            n,
                            instance_seed,
                            solver_seed,
                        )
                    )

    detailed = pd.DataFrame(results)
    summary = aggregate_grover_summary(detailed)
    by_size = aggregate_grover_by_size(detailed)
    return detailed, summary, by_size


def write_suite_outputs(
    output_dir: Path,
    suite_name: str,
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    by_size: pd.DataFrame,
) -> dict[str, str]:
    ensure_dir(output_dir)
    details_path = output_dir / f"{suite_name}_detailed.csv"
    summary_path = output_dir / f"{suite_name}_summary.csv"
    by_size_path = output_dir / f"{suite_name}_by_size.csv"

    detailed.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_size.to_csv(by_size_path, index=False)

    return {
        "detailed_csv": str(details_path),
        "summary_csv": str(summary_path),
        "by_size_csv": str(by_size_path),
    }


def write_markdown_report(
    output_dir: Path,
    config: dict,
    policy_summary: pd.DataFrame | None,
    policy_by_size: pd.DataFrame | None,
    grover_summary: pd.DataFrame | None,
    grover_by_size: pd.DataFrame | None,
    grover_scaling: pd.DataFrame,
) -> Path:
    lines = [
        "# AM-PCVA-OI Article Benchmark Report",
        "",
        "## Configuracao",
        "",
        f"- instance_sizes: {config['instance_sizes']}",
        f"- instance_seeds: {config['instance_seeds']}",
        f"- solver_seeds: {config['solver_seeds']}",
        f"- suites: {config['suites']}",
        f"- grover_scaling_solution_count: {config['grover_scaling_solution_count']}",
        "",
    ]

    if policy_summary is not None:
        lines.extend(
            [
                "## Policies: resumo geral",
                "",
                markdown_table(policy_summary),
                "",
                "## Policies: resumo por tamanho",
                "",
                markdown_table(policy_by_size),
                "",
            ]
        )

    if grover_summary is not None:
        lines.extend(
            [
                "## Grover: resumo geral",
                "",
                markdown_table(grover_summary),
                "",
                "## Grover: resumo por tamanho",
                "",
                markdown_table(grover_by_size),
                "",
            ]
        )

    lines.extend(
        [
            "## Grover: tabela de fundamentacao teorica",
            "",
            "A tabela abaixo compara custo de consulta classico O(M) com a estimativa idealizada de Grover.",
            "",
            markdown_table(grover_scaling, float_digits=4),
            "",
        ]
    )

    report_path = output_dir / "article_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner unificado para gerar benchmarks e tabelas limpas para o artigo."
    )
    parser.add_argument(
        "--suite",
        choices=["all", "policies", "grover", "scaling"],
        default="all",
        help="Conjunto de experimentos a executar.",
    )
    parser.add_argument(
        "--instance-sizes",
        default="30,40,50,60",
        help="Lista de tamanhos de instancia, ex.: 30,40,50,60",
    )
    parser.add_argument(
        "--instance-seeds",
        default="1-10",
        help="Lista ou intervalo de seeds das instancias, ex.: 1-10",
    )
    parser.add_argument(
        "--solver-seeds",
        default="1-5",
        help="Lista ou intervalo de seeds do solver, ex.: 1-5",
    )
    parser.add_argument(
        "--scaling-pool-sizes",
        default="8,16,32,64,128,256",
        help="Tamanhos do pool de candidatos usados na tabela teorica do Grover.",
    )
    parser.add_argument(
        "--scaling-solution-count",
        type=int,
        default=1,
        help="Numero assumido de solucoes marcadas na tabela teorica do Grover.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretorio de saida dos CSVs e do relatorio Markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_sizes = parse_int_list(args.instance_sizes)
    instance_seeds = parse_int_list(args.instance_seeds)
    solver_seeds = parse_int_list(args.solver_seeds)
    scaling_pool_sizes = parse_int_list(args.scaling_pool_sizes)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    config = {
        "instance_sizes": instance_sizes,
        "instance_seeds": instance_seeds,
        "solver_seeds": solver_seeds,
        "suites": args.suite,
        "grover_scaling_solution_count": args.scaling_solution_count,
    }

    manifest = {
        "config": config,
        "outputs": {},
    }

    policy_summary = None
    policy_by_size = None
    grover_summary = None
    grover_by_size = None

    if args.suite in {"all", "policies"}:
        policy_detailed, policy_summary, policy_by_size = run_policy_suite(
            instance_sizes,
            instance_seeds,
            solver_seeds,
        )
        manifest["outputs"]["policies"] = write_suite_outputs(
            output_dir,
            "policies",
            policy_detailed,
            policy_summary,
            policy_by_size,
        )

    if args.suite in {"all", "grover"}:
        grover_detailed, grover_summary, grover_by_size = run_grover_suite(
            instance_sizes,
            instance_seeds,
            solver_seeds,
        )
        manifest["outputs"]["grover"] = write_suite_outputs(
            output_dir,
            "grover",
            grover_detailed,
            grover_summary,
            grover_by_size,
        )

    grover_scaling = build_grover_scaling_table(
        candidate_pool_sizes=scaling_pool_sizes,
        num_solutions=args.scaling_solution_count,
    )
    scaling_path = output_dir / "grover_scaling.csv"
    grover_scaling.to_csv(scaling_path, index=False)
    manifest["outputs"]["grover_scaling_csv"] = str(scaling_path)

    report_path = write_markdown_report(
        output_dir=output_dir,
        config=config,
        policy_summary=policy_summary,
        policy_by_size=policy_by_size,
        grover_summary=grover_summary,
        grover_by_size=grover_by_size,
        grover_scaling=grover_scaling,
    )
    manifest["outputs"]["report_markdown"] = str(report_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[DONE] Article benchmark suite concluido")
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
