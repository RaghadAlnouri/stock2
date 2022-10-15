from typing import Any, Callable, List


# pipeline creator
def create_pipeline(list_functions: List[Callable]) -> Callable[[List[Callable]]]:
    def pipeline(input: Any) -> Any:
        res = input
        for function in list_functions:
            res = function(res)
        return res

    return pipeline

