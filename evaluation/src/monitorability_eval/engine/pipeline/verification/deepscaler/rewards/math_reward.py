"""
Inherited deterministic mathematical-answer scoring. The unused external ORM
backend is intentionally unavailable in this intervention evaluation profile.
"""
from typing import List, Union

from . import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from .math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd, get_multiple_choice_answer
import re
import pdb
ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
choice_pattern = re.compile(r'^[ABCD]$', re.IGNORECASE)

def extract_all_boxed_contents(latex_string):
    """
    Return a list of *all* contents found inside any \boxed{ ... } structure
    in the given LaTeX string, handling nested braces properly.
    """
    pattern = r'\\boxed\s*\{'
    results = []
    start_search_pos = 0
    
    while True:
        # Find the next occurrence of "\boxed{"
        match = re.search(pattern, latex_string[start_search_pos:])
        if not match:
            break  # No more \boxed{ found
        
        # Compute the index right after '\boxed{'
        # match.end() is relative to start_search_pos, so we adjust:
        open_brace_index = start_search_pos + match.end()
        
        # Now collect everything until the matching '}' is found
        brace_count = 1
        i = open_brace_index
        while i < len(latex_string) and brace_count > 0:
            if latex_string[i] == '{':
                brace_count += 1
            elif latex_string[i] == '}':
                brace_count -= 1
            i += 1
        
        # Extract the content (from open_brace_index to the char before i)
        content = latex_string[open_brace_index : i - 1]
        results.append(content)
        
        # Continue searching after this closing brace
        start_search_pos = i
    
    return results


def extract_last_boxed_content(latex_string):
    """
    If you only need the *last* \boxed{...} in the string, 
    we can reuse the above function and return the last one (if any).
    """
    all_boxed = extract_all_boxed_contents(latex_string)
    return all_boxed[-1] if all_boxed else None


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        if self.config.use_math_orm:
            raise ValueError('External ORM is outside the frozen deterministic verifier profile')
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        # Here is format reward. I command it.

        # if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        # model_answer = extract_answer(model_solution)
        # model_answer = extract_last_boxed_content(model_response)
        # model_answer = get_multiple_choice_answer(model_response)

        model_answer = extract_answer(model_response)

        #breakpoint()

        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        
        #breakpoint()

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
