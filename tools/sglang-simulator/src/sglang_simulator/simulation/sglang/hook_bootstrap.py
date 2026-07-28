"""Install HiSim hooks in both the parent and spawned scheduler processes."""

import torch

import sglang_simulator.hook as sglang_simulator_hook
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    hiradix_cache,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    scheduler,
    sgl_kernel_hook,
)

_HOOKS_INSTALLED = False


def install_simulator_hooks() -> None:
    """Install hooks once in the current Python interpreter."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    if not torch.cuda.is_available():
        sglang_simulator_hook.install_module_hooks(
            [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
        )

    sglang_simulator_hook.install_class_hooks(
        [
            scheduler.C_SchedulerHook,
            scheduler.C_SglangPrefillAdderHook,
            scheduler.C_SchedulerRequestReceiver,
            model_runner.C_ModelRunnerHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            hiradix_cache.C_HiRadixCacheHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_MHATokenToKVPoolHostHook,
            mem_pool_host.C_HostKVCacheHook,
            mem_pool_host.C_DeepSeekV4SingleKVPoolHook,
            mem_pool_host.C_DeepSeekV4PagedHostPoolHook,
            mem_pool_host.C_DeepSeekV4StateHostPoolHook,
        ]
    )
    _HOOKS_INSTALLED = True


def run_simulator_scheduler_process(*args, **kwargs):
    """Spawn-safe scheduler entry point which installs HiSim before SGLang imports."""
    install_simulator_hooks()

    # Import only after hook installation. SGLang v0.5.16 forces the spawn start
    # method, so parent-process monkey patches are not inherited by the scheduler.
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
