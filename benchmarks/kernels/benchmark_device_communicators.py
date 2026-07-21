#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for device communicators:
CustomAllreduce (oneshot, twoshot), PyNcclCommunicator,
and SymmMemCommunicator (multimem, two-shot).

for NCCL symmetric memory you need to set the environment variables
NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_USE_NCCL_SYMM_MEM=1, otherwise NCCL does
not use fast NVLS implementation for all reduce.

Usage:
    torchrun --nproc_per_node=<N> benchmark_device_communicators.py [options]

Example:
    torchrun --nproc_per_node=2 benchmark_device_communicators.py
    --sequence-lengths 512 1024 2048 --num-warmup 10 --num-trials 100
"""

import json
import os
import time
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    FlashInferAllReduce,
)
from vllm.distributed.device_communicators.pynccl import (
    PyNcclCommunicator,
    register_nccl_symmetric_ops,
)
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

# Default sequence lengths to benchmark
DEFAULT_SEQUENCE_LENGTHS = [16, 64, 128, 512, 1024, 2048, 4096, 8192]

# Default hidden size and dtype preserve the existing benchmark behavior.
DEFAULT_HIDDEN_SIZE = 8192
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# CUDA graph settings
CUDA_GRAPH_CAPTURE_CYCLES = 10


class CommunicatorBenchmark:
    """Benchmark class for testing device communicators."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        cpu_group: ProcessGroup,
        tensor_elements: list[int],
        dtype: torch.dtype,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.cpu_group = cpu_group
        self.dtype = dtype
        self.max_size_override = max(tensor_elements) * dtype.itemsize + 1

        # Initialize communicators
        self.custom_allreduce = None
        self.pynccl_comm = None
        self.symm_mem_comm = None
        self.symm_mem_comm_multimem = None
        self.symm_mem_comm_two_shot = None
        self.fi_ar_comm = None
        # Communicator variant name -> human-readable reason it could not be
        # initialized/used. Populated for anything unavailable, disabled, or
        # failing to init so reporting never silently drops a backend.
        self.unsupported: dict[str, str] = {}

        self._init_communicators()

    def _init_communicators(self):
        """Initialize all available communicators."""
        try:
            self.custom_allreduce = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                max_size=self.max_size_override,
            )
            if not self.custom_allreduce.disabled:
                logger.info("Rank %s: CustomAllreduce initialized", self.rank)
            else:
                logger.info("Rank %s: CustomAllreduce disabled", self.rank)
                self.custom_allreduce = None
                reason = (
                    "CustomAllreduce reported disabled=True: missing custom-ar "
                    "library, cross-node process group, unsupported world size, "
                    "non-fully-connected multi-GPU topology (PCIe-only with >2 "
                    "GPUs), or no GPU P2P/NVLink support"
                )
                self.unsupported["ca_1stage"] = reason
                self.unsupported["ca_2stage"] = reason
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize CustomAllreduce: %s", self.rank, e
            )
            self.custom_allreduce = None
            reason = f"initialization raised {type(e).__name__}: {e}"
            self.unsupported["ca_1stage"] = reason
            self.unsupported["ca_2stage"] = reason

        try:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group, device=self.device
            )
            if not self.pynccl_comm.disabled:
                logger.info("Rank %s: PyNcclCommunicator initialized", self.rank)
                register_nccl_symmetric_ops(self.pynccl_comm)
            else:
                logger.info("Rank %s: PyNcclCommunicator disabled", self.rank)
                self.pynccl_comm = None
                reason = (
                    "PyNcclCommunicator reported disabled=True: world_size == 1, "
                    "VLLM_DISABLE_PYNCCL is set, or the NCCL/RCCL library failed "
                    "to load"
                )
                self.unsupported["pynccl"] = reason
                self.unsupported["pynccl-symm"] = reason
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize PyNcclCommunicator: %s", self.rank, e
            )
            self.pynccl_comm = None
            reason = f"initialization raised {type(e).__name__}: {e}"
            self.unsupported["pynccl"] = reason
            self.unsupported["pynccl-symm"] = reason

        # Initialize variants for SymmMemCommunicator
        try:
            self.symm_mem_comm_multimem = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
                force_multimem=True,
                max_size_override=self.max_size_override,
            )
            if not self.symm_mem_comm_multimem.disabled:
                logger.info(
                    "Rank %s: SymmMemCommunicator (multimem) initialized", self.rank
                )
            else:
                self.symm_mem_comm_multimem = None
                self.unsupported["symm_mem_multimem"] = (
                    "SymmMemCommunicator(multimem) reported disabled=True: "
                    "symmetric-memory extension unavailable, non-CUDA platform, "
                    "unknown/unsupported device capability or world size, "
                    "buffer rendezvous failed, or multicast operations "
                    "unsupported"
                )
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize SymmMemCommunicator (multimem): %s",
                self.rank,
                e,
            )
            self.symm_mem_comm_multimem = None
            self.unsupported["symm_mem_multimem"] = (
                f"initialization raised {type(e).__name__}: {e}"
            )

        try:
            self.symm_mem_comm_two_shot = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
                force_multimem=False,
                max_size_override=self.max_size_override,
            )
            if not self.symm_mem_comm_two_shot.disabled:
                logger.info(
                    "Rank %s: SymmMemCommunicator (two_shot) initialized", self.rank
                )
            else:
                self.symm_mem_comm_two_shot = None
                self.unsupported["symm_mem_two_shot"] = (
                    "SymmMemCommunicator(two_shot) reported disabled=True: "
                    "symmetric-memory extension unavailable, non-CUDA platform, "
                    "unknown/unsupported device capability or world size, "
                    "buffer rendezvous failed, or multicast operations "
                    "unsupported"
                )
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize SymmMemCommunicator (two_shot): %s",
                self.rank,
                e,
            )
            self.symm_mem_comm_two_shot = None
            self.unsupported["symm_mem_two_shot"] = (
                f"initialization raised {type(e).__name__}: {e}"
            )

        try:
            self.fi_ar_comm = FlashInferAllReduce(
                group=self.cpu_group,
                device=self.device,
            )
            if not self.fi_ar_comm.disabled:
                logger.info("Rank %s: FlashInferAllReduce initialized", self.rank)
            else:
                logger.info("Rank %s: FlashInferAllReduce disabled", self.rank)
                self.fi_ar_comm = None
                reason = (
                    "FlashInferAllReduce reported disabled=True: flashinfer "
                    "library not installed, non-CUDA platform, world_size == 1, "
                    "or no supported allreduce-fusion workspace size for this "
                    "world_size"
                )
                self.unsupported["flashinfer_trtllm"] = reason
                self.unsupported["flashinfer_mnnvl"] = reason
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize FlashInferAllReduce: %s", self.rank, e
            )
            self.fi_ar_comm = None
            reason = f"initialization raised {type(e).__name__}: {e}"
            self.unsupported["flashinfer_trtllm"] = reason
            self.unsupported["flashinfer_mnnvl"] = reason

    def benchmark_allreduce(
        self, tensor_elements: int, num_warmup: int, num_trials: int
    ) -> dict[str, float]:
        """Benchmark allreduce operations for all available communicators."""

        results = {}

        # Define communicators with their benchmark functions
        communicators = []

        if self.custom_allreduce is not None:
            comm = self.custom_allreduce
            # CustomAllreduce one-shot
            communicators.append(
                (
                    "ca_1stage",
                    lambda t, c=comm: c.custom_all_reduce(t),
                    lambda t, c=comm: c.should_custom_ar(t),
                    comm.capture(),
                    {"VLLM_CUSTOM_ALLREDUCE_ALGO": "1stage"},
                    None,  # no destroy function
                )
            )
            # CustomAllreduce two-shot
            communicators.append(
                (
                    "ca_2stage",
                    lambda t, c=comm: c.custom_all_reduce(t),
                    lambda t, c=comm: c.should_custom_ar(t),
                    comm.capture(),
                    {"VLLM_CUSTOM_ALLREDUCE_ALGO": "2stage"},
                    None,  # no destroy function
                )
            )

        if self.pynccl_comm is not None:
            comm = self.pynccl_comm
            communicators.append(
                (
                    "pynccl",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t: True,  # Always available if initialized
                    nullcontext(),
                    {},  # no env variable needed
                    None,  # no destroy function
                )
            )
            communicators.append(
                (
                    "pynccl-symm",
                    lambda t: torch.ops.vllm.all_reduce_symmetric_with_copy(t),
                    lambda t: True,  # Always available if initialized
                    nullcontext(),
                    {},  # no env variable needed
                    None,  # no destroy function
                )
            )

        if self.symm_mem_comm_multimem is not None:
            comm = self.symm_mem_comm_multimem
            communicators.append(
                (
                    "symm_mem_multimem",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_symm_mem(t),
                    nullcontext(),
                    {},  # no env variable needed
                    None,  # no destroy function
                )
            )

        if self.symm_mem_comm_two_shot is not None:
            comm = self.symm_mem_comm_two_shot
            communicators.append(
                (
                    "symm_mem_two_shot",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_symm_mem(t),
                    nullcontext(),
                    {},  # no env variable needed
                    None,  # no destroy function needed
                )
            )

        if self.fi_ar_comm is not None:
            comm = self.fi_ar_comm
            communicators.append(
                (
                    "flashinfer_trtllm",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_fi_ar(t),
                    nullcontext(),
                    {"VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm"},
                    lambda c=comm: c.destroy(),
                )
            )
            communicators.append(
                (
                    "flashinfer_mnnvl",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_fi_ar(t),
                    nullcontext(),
                    {"VLLM_FLASHINFER_ALLREDUCE_BACKEND": "mnnvl"},
                    lambda c=comm: c.destroy(),
                )
            )

        # Benchmark each communicator
        for (
            name,
            allreduce_fn,
            should_use_fn,
            context,
            env_dict,
            destroy_fn,
        ) in communicators:
            # Save original values and apply new environment variables
            saved_env = {key: os.environ.get(key) for key in env_dict}
            for key, value in env_dict.items():
                os.environ[key] = value
            try:
                latency = self.benchmark_allreduce_single(
                    tensor_elements,
                    allreduce_fn,
                    should_use_fn,
                    context,
                    num_warmup,
                    num_trials,
                )
                if latency is not None:
                    results[name] = latency
            finally:
                if destroy_fn is not None:
                    destroy_fn()
                # Restore environment variables to their original state
                for key, original_value in saved_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

        return results

    def benchmark_allreduce_single(
        self,
        tensor_elements: int,
        allreduce_fn: Callable[[torch.Tensor], torch.Tensor | None],
        should_use_fn: Callable[[torch.Tensor], bool],
        context,
        num_warmup: int,
        num_trials: int,
    ) -> float | None:
        """Benchmark an exact one-dimensional tensor with CUDA graph replay."""
        try:
            tensor = torch.randn(tensor_elements, dtype=self.dtype, device=self.device)
            if not should_use_fn(tensor):
                return None

            torch.accelerator.synchronize()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph_input = tensor.clone()

                # Warmup before capture
                for _ in range(3):
                    allreduce_fn(graph_input)

                # Capture the graph using context manager
                with context:
                    graph = torch.cuda.CUDAGraph()
                    graph_pool = torch.cuda.graph_pool_handle()
                    set_graph_pool_id(graph_pool)
                    with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
                        for _ in range(CUDA_GRAPH_CAPTURE_CYCLES):
                            allreduce_fn(graph_input)

            torch.accelerator.synchronize()
            for _ in range(num_warmup):
                graph.replay()
            torch.accelerator.synchronize()

            torch.accelerator.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_trials):
                graph.replay()
            torch.accelerator.synchronize()

            end_time = time.perf_counter()

            # Convert to ms and divide by CUDA_GRAPH_CAPTURE_CYCLES
            return (
                (end_time - start_time) / num_trials / CUDA_GRAPH_CAPTURE_CYCLES * 1000
            )

        except Exception as e:
            logger.error("CUDA graph benchmark failed: %s", e)
            raise RuntimeError(
                f"CUDA graph benchmark failed for communicator: {e}"
            ) from e


def _calculate_speedup_info(comm_results: dict[str, float]) -> str:
    """Calculate speedup information for a single tensor size."""
    if not comm_results:
        return "N/A"

    # Find the fastest communicator
    fastest_comm = min(comm_results.keys(), key=lambda k: comm_results[k])
    fastest_time = comm_results[fastest_comm]

    # Calculate speedup vs PyNccl if available
    if "pynccl" in comm_results:
        pynccl_time = comm_results["pynccl"]
        speedup = pynccl_time / fastest_time
        return f"{fastest_comm} ({speedup:.2f}x)"
    else:
        return f"{fastest_comm} (N/A)"


def _flatten(values: list[list[int]] | None) -> list[int] | None:
    if values is None:
        return None
    return [value for group in values for value in group]


def resolve_tensor_elements(args) -> tuple[list[int], str]:
    """Resolve exactly one sizing mode into exact tensor element counts."""
    num_elements = _flatten(args.num_elements)
    message_size_bytes = _flatten(args.message_size_bytes)
    if num_elements is not None and message_size_bytes is not None:
        raise ValueError(
            "--num-elements and --message-size-bytes are mutually exclusive"
        )
    dtype = DTYPES[args.dtype]
    if num_elements is not None:
        if any(value <= 0 for value in num_elements):
            raise ValueError("--num-elements values must be positive")
        return num_elements, "elements"
    if message_size_bytes is not None:
        if any(value <= 0 or value % dtype.itemsize for value in message_size_bytes):
            raise ValueError(
                "--message-size-bytes values must be positive dtype multiples"
            )
        return [value // dtype.itemsize for value in message_size_bytes], "bytes"
    if any(value <= 0 for value in args.sequence_lengths):
        raise ValueError("--sequence-lengths values must be positive")
    return [
        length * args.hidden_size for length in args.sequence_lengths
    ], "sequence_lengths"


def print_results(
    results: dict[int, dict[str, float]],
    tensor_elements: list[int],
    world_size: int,
    dtype: torch.dtype,
    hidden_size: int,
    size_mode: str,
    unsupported: dict[str, str] | None = None,
) -> None:
    """Print timings while retaining exact byte and element sizing."""
    print(f"\n{'=' * 130}")
    print("Device Communicator Benchmark Results")
    print(f"World Size: {world_size}, Data Type: {dtype}, Hidden Size: {hidden_size}")
    print(f"{'=' * 130}")
    all_comms = sorted(
        {comm for size_results in results.values() for comm in size_results}
    )
    header = f"{'Tensor Elements':<20}{'Tensor Size':<15}"
    for comm in all_comms:
        header += f"{comm:<20}"
    header += f"{'Best (Speedup vs PyNccl)':<30}"
    print(header)
    print("-" * len(header))
    for elements in tensor_elements:
        if elements not in results:
            continue
        tensor_bytes = elements * dtype.itemsize
        tensor_size_str = f"{tensor_bytes / (1024 * 1024):.2f} MB"
        row = f"{elements:<20}{tensor_size_str:<15}"
        for comm in all_comms:
            row += (
                f"{results[elements][comm]:<20.3f}"
                if comm in results[elements]
                else f"{'N/A':<20}"
            )
        row += f"{_calculate_speedup_info(results[elements]):<30}"
        print(row)
    print(f"{'=' * 130}")
    print(
        f"All times are milliseconds per allreduce operation (sizing mode: {size_mode})"
    )
    if unsupported:
        print(f"\n{'-' * 130}")
        print("Unsupported communicators (excluded from the table above):")
        for name in sorted(unsupported):
            print(f"  {name}: UNSUPPORTED - {unsupported[name]}")


def main() -> None:
    parser = FlexibleArgumentParser(description="Benchmark device communicators")
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Legacy tensor shape sizes: seq_len x hidden_size.",
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--num-elements",
        type=int,
        nargs="+",
        action="append",
        help="Exact one-dimensional tensor element count; repeatable.",
    )
    parser.add_argument(
        "--message-size-bytes",
        type=int,
        nargs="+",
        action="append",
        help="Exact tensor byte size; repeatable and divisible by dtype size.",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-trials", type=int, default=50, help="Number of benchmark trials"
    )
    parser.add_argument("--output-json", type=str, help="Output results to JSON file")
    args = parser.parse_args()
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    try:
        tensor_elements, size_mode = resolve_tensor_elements(args)
    except ValueError as error:
        parser.error(str(error))
    dtype = DTYPES[args.dtype]

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    cpu_group = dist.new_group(backend="gloo")
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    benchmark = CommunicatorBenchmark(
        rank, world_size, device, cpu_group, tensor_elements, dtype
    )
    all_results: dict[int, dict[str, float]] = {}
    for elements in tensor_elements:
        if rank == 0:
            logger.info(
                "Benchmarking %s elements (%s bytes, dtype=%s)",
                elements,
                elements * dtype.itemsize,
                args.dtype,
            )
        all_results[elements] = benchmark.benchmark_allreduce(
            tensor_elements=elements,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )
        dist.barrier()
    if rank == 0:
        print_results(
            all_results,
            tensor_elements,
            world_size,
            dtype,
            args.hidden_size,
            size_mode,
            benchmark.unsupported,
        )
        if args.output_json:
            output_data = {
                "world_size": world_size,
                "dtype": args.dtype,
                "hidden_size": args.hidden_size,
                "size_mode": size_mode,
                "tensor_elements": tensor_elements,
                "message_size_bytes": [
                    elements * dtype.itemsize for elements in tensor_elements
                ],
                "num_warmup": args.num_warmup,
                "num_trials": args.num_trials,
                "cuda_graph_capture_cycles": CUDA_GRAPH_CAPTURE_CYCLES,
                "unsupported": benchmark.unsupported,
                "results": {
                    str(elements): {
                        "timings": comm_results,
                        "speedup_info": _calculate_speedup_info(comm_results),
                    }
                    for elements, comm_results in all_results.items()
                },
            }
            with open(args.output_json, "w") as file:
                json.dump(output_data, file, indent=2)
            logger.info("Results saved to %s", args.output_json)
    if cpu_group != dist.group.WORLD:
        dist.destroy_process_group(cpu_group)


if __name__ == "__main__":
    main()
