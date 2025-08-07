import json
import copy
from typing import Dict, Any, List, Optional
from tag_common import TagCommon


class DifficultyDataset:
    """难度评估数据集，负责处理原始数据并输出模型可用的数据格式"""
    
    def __init__(self):
        """初始化数据集"""
        self.task_type = "difficulty"
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSONL格式的数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据列表
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON line: {e}")
        return data
    
    def save_data(self, data: List[Dict[str, Any]], file_path: str):
        """
        保存数据到JSONL文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def append_data(self, data: List[Dict[str, Any]], file_path: str):
        """
        追加数据到JSONL文件
        
        Args:
            data: 要追加的数据
            file_path: 保存路径
        """
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def extract_statement(self, item: Dict[str, Any]) -> str:
        """
        从数据项中提取数学问题描述
        
        Args:
            item: 数据项
            
        Returns:
            数学问题描述
        """
        return item.get("natural_language_statement", "").strip()
    
    def is_complete(self, item: Dict[str, Any]) -> bool:
        """
        检查数据项是否完整（已包含难度信息）
        
        Args:
            item: 数据项
            
        Returns:
            是否完整
        """
        return bool(item.get("difficulty")) and bool(item.get("difficulty_rationale"))
    
    def filter_incomplete(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤出不完整的数据项
        
        Args:
            data: 原始数据
            
        Returns:
            不完整的数据项列表
        """
        return [item for item in data if not self.is_complete(item)]
    
    def filter_complete(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤出完整的数据项
        
        Args:
            data: 原始数据
            
        Returns:
            完整的数据项列表
        """
        return [item for item in data if self.is_complete(item)]
    
    def update_item_with_prediction(self, item: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        用预测结果更新数据项
        
        Args:
            item: 原始数据项
            prediction: 预测结果
            
        Returns:
            更新后的数据项
        """
        updated_item = copy.deepcopy(item)
        
        if prediction.get("status") == "success" and prediction.get("data"):
            data = prediction["data"]
            updated_item.update({
                "difficulty": data.get("Difficulty"),
                "difficulty_rationale": data.get("Rationale"),
                "difficulty_response": prediction.get("response")
            })
        
        return updated_item
    
    def prepare_for_model(self, item: Dict[str, Any]) -> str:
        """
        为模型准备输入数据
        
        Args:
            item: 数据项
            
        Returns:
            模型输入（数学问题描述）
        """
        return self.extract_statement(item)
    
    def build_prompt(self, statement: str) -> str:
        """
        构建难度评估提示
        
        Args:
            statement: 数学问题描述
            
        Returns:
            构建的提示
        """
        return TagCommon.build_difficulty_annotation_prompt(statement)
    
    def get_expected_keys(self) -> List[str]:
        """
        获取期望的JSON键
        
        Returns:
            期望的键列表
        """
        return ["Difficulty", "Rationale"]
    
    def get_validation_function(self):
        """
        获取验证函数
        
        Returns:
            验证函数
        """
        return TagCommon.validate_difficulty_response
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        验证预测结果是否有效
        
        Args:
            prediction: 预测结果
            
        Returns:
            是否有效
        """
        if prediction.get("status") != "success":
            return False
        
        data = prediction.get("data")
        if not data:
            return False
        
        return TagCommon.validate_difficulty_response(data)
    
    def create_batch(self, data: List[Dict[str, Any]], batch_size: int = 1) -> List[List[Dict[str, Any]]]:
        """
        创建批次数据
        
        Args:
            data: 原始数据
            batch_size: 批次大小
            
        Returns:
            批次数据列表
        """
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def merge_results(self, original_data: List[Dict[str, Any]], 
                     processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并原始数据和处理结果
        
        Args:
            original_data: 原始数据
            processed_data: 处理后的数据
            
        Returns:
            合并后的数据
        """
        # 创建原始数据的映射
        original_map = {}
        for item in original_data:
            statement = self.extract_statement(item)
            if statement:
                original_map[statement] = item
        
        # 更新处理后的数据
        for processed_item in processed_data:
            statement = self.extract_statement(processed_item)
            if statement in original_map:
                original_map[statement].update(processed_item)
        
        return list(original_map.values())
    
    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            data: 数据列表
            
        Returns:
            统计信息
        """
        total = len(data)
        complete = len(self.filter_complete(data))
        incomplete = total - complete
        
        difficulties = [item.get("difficulty") for item in data if item.get("difficulty") is not None]
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        
        stats = {
            "total": total,
            "complete": complete,
            "incomplete": incomplete,
            "completion_rate": complete / total if total > 0 else 0,
            "avg_difficulty": avg_difficulty,
            "difficulty_distribution": self._get_difficulty_distribution(difficulties)
        }
        
        return stats
    
    def _get_difficulty_distribution(self, difficulties: List[float]) -> Dict[str, int]:
        """
        获取难度分布
        
        Args:
            difficulties: 难度列表
            
        Returns:
            难度分布
        """
        distribution = {}
        for diff in difficulties:
            range_key = f"{int(diff)}-{int(diff)+1}"
            distribution[range_key] = distribution.get(range_key, 0) + 1
        return distribution


# 提示模板（从TagCommon中提取，便于自定义）
DIFFICULTY_PROMPT_TEMPLATE = """
I am working with natural language statements of advanced mathematics problems, which are intended to be later formalized in Lean.
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
```
"""
