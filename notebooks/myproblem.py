from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_constr=0, xl=-5, xu=5,
                         elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = (x ** 2).sum()