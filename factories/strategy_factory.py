from strategies.logits_distillation import LogitsDistillation

class StrategyFactory:
    @staticmethod
    def create_strategy(teacher_adapter, student_adapter):
        return LogitsDistillation()