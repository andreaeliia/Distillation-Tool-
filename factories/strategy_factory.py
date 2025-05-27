from strategies.logits_distillation import LogitsDistillation
from strategies.hard_soft_distillation import HardSoftDistillation
from strategies.chunck_hard_soft_distillation import ChunkHardSoftDistillation

class StrategyFactory:
    @staticmethod
    def create_strategy(teacher_adapter, student_adapter, strategy_name, **kwargs):
        if strategy_name == "logits":
            return LogitsDistillation()
        elif strategy_name == "hard_soft":
            return HardSoftDistillation(**kwargs)
        elif strategy_name == "chunked":
            return ChunkHardSoftDistillation(**kwargs)
        else:
            raise ValueError(f"Strategia '{strategy_name}' non supportata.")
