from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self):
        return False

    def supports_generation_cuda_graph(self) -> bool:
        return False
