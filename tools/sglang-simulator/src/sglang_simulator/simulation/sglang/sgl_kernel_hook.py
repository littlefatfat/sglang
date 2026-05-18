from sglang_simulator.hook import BaseHook


class M_SGLangKernelLoadUtilHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sgl_kernel.load_utils"

    @classmethod
    def hook(cls, target):
        def override_load_architecture_specific_ops(*args, **kwargs):
            """
            ImportError:
            [sgl_kernel] CRITICAL: Could not load any common_ops library!
            """
            pass

        target._load_architecture_specific_ops = override_load_architecture_specific_ops


class M_SGLangCommonHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sglang.srt.utils.common"

    @classmethod
    def hook(cls, target):
        def override_support_triton(backend: str) -> bool:
            # print(f"[override_support_triton] {backend=}")
            return backend not in ["torch_native", "intel_amx", "compressed", "dsv4"]
    
        target.support_triton = override_support_triton