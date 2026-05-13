from __future__ import annotations

import argparse
import math
import os
from typing import Sequence


DEFAULT_RESOURCE_ID = os.getenv("AZURE_QUANTUM_RESOURCE_ID", "")
DEFAULT_TARGET = os.getenv("AZURE_QUANTUM_TARGET", "")
DEFAULT_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
DEFAULT_AUTH_DEBUG = os.getenv("AZURE_QUANTUM_AUTH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def validate_good_indices(good_indices: Sequence[int], num_items: int) -> list[int]:
    unique = sorted(set(good_indices))
    if not unique:
        raise ValueError("Informe ao menos um indice bom em --good-indices.")
    if unique[0] < 0:
        raise ValueError("Os indices bons devem ser inteiros nao negativos.")
    if unique[-1] >= num_items:
        raise ValueError("Todos os indices bons devem ser menores que --num-items.")
    return unique


def build_phase_oracle(good_states: Sequence[str], num_qubits: int):
    from qiskit import QuantumCircuit

    oracle = QuantumCircuit(num_qubits, name="oracle")
    target_qubit = num_qubits - 1
    control_qubits = list(range(num_qubits - 1))

    for state in good_states:
        reversed_state = state[::-1]
        zero_positions = [idx for idx, bit in enumerate(reversed_state) if bit == "0"]

        for qubit in zero_positions:
            oracle.x(qubit)

        if num_qubits == 1:
            oracle.z(0)
        else:
            oracle.h(target_qubit)
            oracle.mcx(control_qubits, target_qubit)
            oracle.h(target_qubit)

        for qubit in reversed(zero_positions):
            oracle.x(qubit)

    return oracle


def build_diffuser(num_qubits: int):
    from qiskit import QuantumCircuit

    diffuser = QuantumCircuit(num_qubits, name="diffuser")
    target_qubit = num_qubits - 1
    control_qubits = list(range(num_qubits - 1))

    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))

    if num_qubits == 1:
        diffuser.z(0)
    else:
        diffuser.h(target_qubit)
        diffuser.mcx(control_qubits, target_qubit)
        diffuser.h(target_qubit)

    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    return diffuser


def grover_iterations(num_qubits: int, num_solutions: int) -> int:
    search_space = 2**num_qubits
    estimate = math.floor((math.pi / 4.0) * math.sqrt(search_space / max(num_solutions, 1)))
    return max(1, estimate)


def build_grover_circuit(num_items: int, good_indices: Sequence[int], iterations: int):
    from qiskit import QuantumCircuit

    num_qubits = max(1, math.ceil(math.log2(max(num_items, 2))))
    good_states = [format(idx, f"0{num_qubits}b") for idx in good_indices]
    oracle = build_phase_oracle(good_states, num_qubits)
    diffuser = build_diffuser(num_qubits)

    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.name = "AM-PCVA-OI Grover Azure Runner"
    circuit.h(range(num_qubits))

    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True)

    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit, num_qubits, good_states


def get_counts(result, circuit):
    try:
        return result.get_counts(circuit)
    except Exception:
        pass

    try:
        return result.get_counts()
    except Exception:
        pass

    raw_counts = getattr(result, "results", None)
    if raw_counts:
        try:
            return dict(result.results[0].data.counts)
        except Exception:
            pass

    raise RuntimeError("Nao foi possivel extrair as contagens do resultado do job.")


def format_counts(counts: dict[str, int]) -> str:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{state}:{count}" for state, count in ordered)


def run_azure_grover_job(
    resource_id: str,
    target: str,
    tenant_id: str | None,
    auth_debug: bool,
    num_items: int,
    good_indices: Sequence[int],
    shots: int,
    iterations: int | None,
    job_name: str,
) -> dict:
    validated_good_indices = validate_good_indices(good_indices, num_items)
    if iterations is None:
        num_qubits = max(1, math.ceil(math.log2(max(num_items, 2))))
        iterations = grover_iterations(num_qubits, len(validated_good_indices))

    circuit, num_qubits, good_states = build_grover_circuit(num_items, validated_good_indices, iterations)

    from qiskit import transpile

    _, backend = resolve_azure_backend(resource_id, target, tenant_id, auth_debug=auth_debug)
    transpiled = transpile(circuit, backend=backend)
    try:
        job = backend.run(transpiled, shots=shots, job_name=job_name)
    except TypeError:
        job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = get_counts(result, transpiled)

    top_state, top_count = max(counts.items(), key=lambda item: item[1])
    top_index = int(top_state, 2)
    is_valid_item = top_index < num_items
    is_marked_solution = top_index in validated_good_indices

    return {
        "job_id": job.job_id(),
        "target": target,
        "counts": counts,
        "formatted_counts": format_counts(counts),
        "top_state": top_state,
        "top_index": top_index,
        "top_count": top_count,
        "valid_item": is_valid_item,
        "marked_solution": is_marked_solution,
        "num_qubits": num_qubits,
        "good_states": good_states,
        "good_indices": list(validated_good_indices),
        "iterations": iterations,
        "shots": shots,
    }


class DebugChainedCredential:
    def __init__(
        self,
        entries: Sequence[tuple[str, object]],
        tenant_id: str | None = None,
        auth_debug: bool = False,
    ):
        self._entries = list(entries)
        self._tenant_id = (tenant_id or "").strip() or "<default>"
        self._auth_debug = auth_debug
        self._selected_label: str | None = None
        self._selected_credential = None

    def get_token(self, *scopes, **kwargs):
        if self._selected_credential is not None:
            return self._selected_credential.get_token(*scopes, **kwargs)

        errors: list[str] = []
        for label, credential in self._entries:
            try:
                token = credential.get_token(*scopes, **kwargs)
                self._selected_label = label
                self._selected_credential = credential
                if self._auth_debug:
                    scope_list = ",".join(scopes) if scopes else "<none>"
                    print(
                        f"[azure-auth] credential={label} tenant_id={self._tenant_id} scopes={scope_list}",
                        flush=True,
                    )
                return token
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        message = "\n".join(errors) if errors else "Nenhuma credencial foi tentada."
        raise RuntimeError(f"Falha ao obter token Azure.\n{message}")

    def close(self) -> None:
        for _, credential in self._entries:
            close = getattr(credential, "close", None)
            if callable(close):
                close()


def build_azure_credential(tenant_id: str | None = None, auth_debug: bool = False):
    from azure.identity import (
        AzureCliCredential,
        AzureDeveloperCliCredential,
        AzurePowerShellCredential,
        DefaultAzureCredential,
    )

    normalized_tenant_id = (tenant_id or "").strip()
    if not normalized_tenant_id:
        return DebugChainedCredential(
            [
                ("DefaultAzureCredential", DefaultAzureCredential(exclude_interactive_browser_credential=False)),
            ],
            auth_debug=auth_debug,
        )

    return DebugChainedCredential(
        [
            ("AzureCliCredential", AzureCliCredential(tenant_id=normalized_tenant_id)),
            ("AzurePowerShellCredential", AzurePowerShellCredential(tenant_id=normalized_tenant_id)),
            ("AzureDeveloperCliCredential", AzureDeveloperCliCredential(tenant_id=normalized_tenant_id)),
            (
                "DefaultAzureCredential",
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_interactive_browser_credential=False,
                    interactive_browser_tenant_id=normalized_tenant_id,
                    shared_cache_tenant_id=normalized_tenant_id,
                    visual_studio_code_tenant_id=normalized_tenant_id,
                    workload_identity_tenant_id=normalized_tenant_id,
                ),
            ),
        ],
        tenant_id=normalized_tenant_id,
        auth_debug=auth_debug,
    )


def resolve_azure_backend(
    resource_id: str,
    target: str,
    tenant_id: str | None = None,
    auth_debug: bool = False,
):
    from qdk.azure import Workspace
    from qdk.azure.qiskit import AzureQuantumProvider

    workspace = Workspace(
        resource_id=resource_id,
        credential=build_azure_credential(tenant_id=tenant_id, auth_debug=auth_debug),
    )
    provider = AzureQuantumProvider(workspace)
    backend = provider.get_backend(target)
    return provider, backend


def list_targets(resource_id: str, tenant_id: str | None = None, auth_debug: bool = False) -> None:
    from qdk.azure import Workspace
    from qdk.azure.qiskit import AzureQuantumProvider

    workspace = Workspace(
        resource_id=resource_id,
        credential=build_azure_credential(tenant_id=tenant_id, auth_debug=auth_debug),
    )
    provider = AzureQuantumProvider(workspace)
    for backend in provider.backends():
        print(backend.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submete um circuito de Grover isolado ao Azure Quantum para testar a subrotina quantica."
    )
    parser.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID, help="Resource ID do workspace Azure Quantum.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Nome do target Azure Quantum.")
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Tenant ID usado para obter o token Azure AD. Se omitido, usa a cadeia padrao de credenciais.",
    )
    parser.add_argument(
        "--auth-debug",
        action="store_true",
        default=DEFAULT_AUTH_DEBUG,
        help="Imprime qual credencial Azure foi selecionada para obter o token.",
    )
    parser.add_argument(
        "--good-indices",
        default="1,3",
        help="Indices marcados como bons no espaco de busca, separados por virgula.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=8,
        help="Numero de itens reais no espaco de busca. O circuito usa ceil(log2(num_items)) qubits.",
    )
    parser.add_argument("--shots", type=int, default=1000, help="Numero de shots do job.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Numero de iteracoes de Grover. Se omitido, usa a estimativa teorica padrao.",
    )
    parser.add_argument(
        "--job-name",
        default="AM-PCVA-OI Grover Azure Runner",
        help="Nome do job enviado ao target.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Lista os backends disponiveis no workspace e encerra.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nao envia job. Apenas monta o circuito e imprime os estados bons.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    try:
        import qiskit  # noqa: F401
        import qdk  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Instale as dependencias opcionais com: pip install --upgrade \"qdk[azure,qiskit]\""
        ) from exc


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()

    if args.list_targets:
        if not args.resource_id:
            raise ValueError("Informe --resource-id ou defina AZURE_QUANTUM_RESOURCE_ID para listar os targets.")
        list_targets(args.resource_id, args.tenant_id, auth_debug=args.auth_debug)
        return

    if args.num_items < 2:
        raise ValueError("--num-items deve ser maior ou igual a 2.")

    good_indices = validate_good_indices(parse_csv_ints(args.good_indices), args.num_items)
    iterations = args.iterations
    if iterations is None:
        num_qubits = max(1, math.ceil(math.log2(max(args.num_items, 2))))
        iterations = grover_iterations(num_qubits, len(good_indices))
    else:
        num_qubits = max(1, math.ceil(math.log2(max(args.num_items, 2))))
    _, _, good_states = build_grover_circuit(args.num_items, good_indices, iterations)

    print("=" * 72)
    print("[AZURE QUANTUM RUNNER - GROVER]")
    print(f"num_items={args.num_items}")
    print(f"num_qubits={num_qubits}")
    print(f"good_indices={good_indices}")
    print(f"good_states={good_states}")
    print(f"iterations={iterations}")
    print(f"shots={args.shots}")
    print("=" * 72)

    if args.dry_run:
        print(circuit.draw(output="text"))
        return

    if not args.resource_id:
        raise ValueError("Informe --resource-id ou defina AZURE_QUANTUM_RESOURCE_ID.")
    if not args.target:
        raise ValueError("Informe --target ou defina AZURE_QUANTUM_TARGET.")

    run_info = run_azure_grover_job(
        resource_id=args.resource_id,
        target=args.target,
        tenant_id=args.tenant_id,
        auth_debug=args.auth_debug,
        num_items=args.num_items,
        good_indices=good_indices,
        shots=args.shots,
        iterations=iterations,
        job_name=args.job_name,
    )

    print(f"job_id={run_info['job_id']}")
    print(f"target={run_info['target']}")
    print(f"counts={run_info['formatted_counts']}")
    print(f"top_state={run_info['top_state']}")
    print(f"top_index={run_info['top_index']}")
    print(f"top_count={run_info['top_count']}")
    print(f"valid_item={run_info['valid_item']}")
    print(f"marked_solution={run_info['marked_solution']}")


if __name__ == "__main__":
    main()
