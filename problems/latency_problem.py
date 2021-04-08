from combined_problem import CombinedProblem


class LatencyProblem(CombinedProblem):
    def __init__(self, instances_df, system_parameters, *args, **kwargs):
        super().__init__(instances_df, system_parameters, *args, **kwargs)
