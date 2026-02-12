from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus

from evalplus.data import get_human_eval_plus

problems_humaneval = get_human_eval_plus()

first_key_h = sorted(problems_humaneval.keys())[0]
print("\n=== HumanEval ===\n")
print("Task ID:", first_key_h)
print("\n=== PROMPT ===\n")
print(problems_humaneval[first_key_h]["prompt"])


from evalplus.data import get_mbpp_plus

problems_mbpp = get_mbpp_plus()

first_key_m = sorted(problems_mbpp.keys())[0]
print("\n=== MBPP ===\n")
print("Task ID:", first_key_m)
print("\n=== PROMPT ===\n")
print(problems_mbpp[first_key_m]["prompt"])