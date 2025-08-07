import json
import re
import time
from typing import Callable, Optional, Dict, Any


class TagCommon:
    """公共标记功能模块，包含提示构建、JSON解析和重试机制"""

    @staticmethod
    def build_difficulty_annotation_prompt(statement: str) -> str:
        """构建难度评估提示"""
        return f"""I am working with natural language statements of advanced mathematics problems, which are intended to be later formalized in Lean.
Before that, we aim to assess the **intrinsic difficulty** of the problem in its current informal (natural language) form.

# OBJECTIVE

Assign a **difficulty score** to the problem, on a scale from 0 to 10.

Your rating should reflect the mathematical reasoning required to solve the problem — including the level of abstraction, creativity, number of steps, and familiarity with advanced techniques.

# DIFFICULTY REFERENCE

## Examples for difficulty levels

For reference, here are problems from each of the difficulty levels 1-10:

1: How many integer values of x satisfy |x| < 3π? (2021 Spring AMC 10B, Problem 1)

1.5: A number is called flippy if its digits alternate between two distinct digits. For example, 2020 and 37373 are flippy, but 3883 and 123123 are not. How many five-digit flippy numbers are divisible by 15? (2020 AMC 8, Problem 19)

2: A fair 6-sided die is repeatedly rolled until an odd number appears. What is the probability that every even number appears at least once before the first occurrence of an odd number? (2021 Spring AMC 10B, Problem 18)

2.5: A, B, C are three piles of rocks. The mean weight of the rocks in A is 40 pounds, the mean weight of the rocks in B is 50 pounds, the mean weight of the rocks in the combined piles A and B is 43 pounds, and the mean weight of the rocks in the combined piles A and C is 44 pounds. What is the greatest possible integer value for the mean in pounds of the rocks in the combined piles B and C? (2013 AMC 12A, Problem 16)

3: Triangle ABC with AB = 50 and AC = 10 has area 120. Let D be the midpoint of AB, and let E be the midpoint of AC. The angle bisector of ∠BAC intersects DE and BC at F and G, respectively. What is the area of quadrilateral FDBG? (2018 AMC 10A, Problem 24)

3.5: Find the number of integer values of k in the closed interval [−500, 500] for which the equation log(kx) = 2 log(x + 2) has exactly one real solution. (2017 AIME II, Problem 7)

4: Define a sequence recursively by x0 = 5 and

xn+1 = xn^2 + 5xn + 4 / xn + 6

for all nonnegative integers n. Let m be the least positive integer such that

xm ≤ 4 + 1 / 2^20.

In which of the following intervals does m lie? (A) [9, 26] (B) [27, 80] (C) [81, 242] (D) [243, 728] (E) [729, ∞) (2019 AMC 10B, Problem 24 and 2019 AMC 12B, Problem 22)

4.5: Find, with proof, all positive integers n for which 2n + 12n + 2011n is a perfect square. (USAJMO 2011/1)

5: Find all triples (a, b, c) of real numbers such that the following system holds:

a + b + c = 1 / a + 1 / b + 1 / c ,

a^2 + b^2 + c^2 = 1 / a^2 + 1 / b^2 + 1 / c^2.(JBMO 2020/1)

5.5: Triangle ABC has ∠BAC = 60◦, ∠CBA ≤ 90◦, BC = 1, and AC ≥ AB. Let H, I, and O be the orthocenter, incenter, and circumcenter of △ABC, respectively. Assume that the area of pentagon BCOIH is the maximum possible. What is ∠CBA? (2011 AMC 12A, Problem 25)

6: Let △ABC be an acute triangle with circumcircle ω, and let H be the intersection of the altitudes of △ABC. Suppose the tangent to the circumcircle of △HBC at H intersects ω at points X and Y with HA = 3, HX = 2, and HY = 6. The area of △ABC can be written in the form m√n, where m and n are positive integers, and n is not divisible by the square of any prime. Find m + n. (2020 AIME I, Problem 15)

6.5: Rectangles BCC1B2, CAA1C2, and ABB1A2 are erected outside an acute triangle ABC. Suppose that ∠BC1C + ∠CA1A + ∠AB1B = 180◦.

Prove that lines B1C2, C1A2, and A1B2 are concurrent. (USAMO 2021/1, USAJMO 2021/2)

7: We say that a finite set S in the plane is balanced if, for any two different points A, B in S, there is a point C in S such that AC = BC. We say that S is centre-free if for any three points A, B, C in S, there is no point P in S such that PA = PB = PC. Show that for all integers n ≥ 3, there exists a balanced set consisting of n points. Determine all integers n ≥ 3 for which there exists a balanced centre-free set consisting of n points. (IMO 2015/1)

7.5: Let Z be the set of integers. Find all functions f : Z → Z such that

x f (2f(y) - x) + y2 f (2x - f (y)) = f (x)2

x + f (y f (y))

for all x, y ∈ Z with x ̸= 0. (USAMO 2014/2)

8: For each positive integer n, the Bank of Cape Town issues coins of denomination 1/n. Given a finite collection of such coins (of not necessarily different denominations) with total value at most most 99 + 1/2, prove that it is possible to split this collection into 100 or fewer groups, such that each group has total value at most 1. (IMO 2014/5)

8.5: Let I be the incentre of acute triangle ABC with AB ̸= AC. The incircle ω of ABC is tangent to sides BC, CA, and AB at D, E, and F, respectively. The line through D perpendicular to EF meets ω at R. Line AR meets ω again at P. The circumcircles of triangle PCE and PBF meet again at Q. Prove that lines DI and PQ meet on the line through A perpendicular to AI. (IMO 2019/6)

9: Let k be a positive integer and let S be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of S around the circle such that the product of any two neighbors is of the form x2 + x + k for some positive integer x. (IMO 2022/3)

9.5: An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from 1 to 10.

4

2 6

5 7 1

8 3 10 9

Does there exist an anti-Pascal triangle with 2018 rows which contains every integer from 1 to 1 + 2 + 3 + · · · + 2018? (IMO 2018/3)

10: Prove that there exists a positive constant c such that the following statement is true: Consider an integer n > 1, and a set S of n points in the plane such that the distance between any two different points in S is at least 1. It follows that there is a line ℓ separating S such that the distance from any point of S to ℓ is at least cn^(-1/3).

# OBJECTIVE #
1. Summarize the math problem in a brief sentence, describing the concepts involved in the math problem.
2. Based on the source of the given problem, as well as the difficulty of the problems referenced in these materials and the solution to the current problem, please provide an overall difficulty score for the current problem. The score should be a number between 1 and 10, with increments of 0.5, and should align perfectly with the materials.

# STYLE #
Data report.

# TONE #
Professional, scientific.

# AUDIENCE #
Students. Enable them to better understand the difficulty of the math problems.

# RESPONSE: MARKDOWN REPORT #
## Summarization  
[Summarize the math problem in a brief paragraph.]

## Difficulty  
[Rate the difficulty of the math problem and give the reason.]

# ATTENTION #
- Add "=== report over ===" at the end of the report.


# INPUT #
Below is the original natural language math problem statement:

<statement>
{statement}
</statement>

# OUTPUT FORMAT #

You must respond with a JSON object:

```json
{{
  "Difficulty": float (between 0 and 10),
  "Rationale": "Explain your score in 1–3 sentences. Mention structural elements or comparison to benchmark problems."
}}
```"""

    @staticmethod
    def extract_json_from_response(text: str, expected_keys: Optional[list] = None) -> str:
        """
        提取字符串中的 JSON 内容，并验证是否包含 expected_keys 中的所有字段。
        expected_keys 为 None 时默认只提取。
        """
        # 尝试直接解析
        try:
            json.loads(text)
            if expected_keys:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in expected_keys:
                        if key not in data:
                            raise ValueError(f"Missing expected key: {key}")
            return text
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # 清理可能的额外字符
                    cleaned = re.sub(r'[^\x20-\x7E]', '', match)
                    data = json.loads(cleaned)
                    
                    # 验证期望的键
                    if expected_keys and isinstance(data, dict):
                        for key in expected_keys:
                            if key not in data:
                                continue
                        return cleaned
                    elif not expected_keys:
                        return cleaned
                except json.JSONDecodeError:
                    continue

        raise ValueError("No valid JSON found in response")

    @staticmethod
    def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.5,
        backoff_factor: float = 1.5
    ) -> Any:
        """
        通用重试机制，支持指数退避
        """
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (backoff_factor ** attempt))

    @staticmethod
    def validate_difficulty_response(data: Dict[str, Any]) -> bool:
        """验证难度响应数据"""
        return isinstance(data, dict) and "Difficulty" in data
