from sglang_simulator.hook import BaseHook


class C_ModelConfigHook(BaseHook):

    HOOK_CLASS_NAME = "ModelConfig"
    HOOK_MODULE_NAME = "sglang.srt.configs.model_config"

    @classmethod
    def hook(cls, target):

        original_derive_hybrid_model = target._derive_hybrid_model

        def wrapped_derive_hybrid_model(self, *args, **kwargs):
            ret = original_derive_hybrid_model(self, *args, **kwargs)
            if self.is_hybrid_swa:
                setattr(self, "is_hybrid_swa_backup", True)
                # Disable hybrid SWA to simplify memory pool hijacking.
                self.is_hybrid_swa = False
            return ret

        target._derive_hybrid_model = wrapped_derive_hybrid_model
