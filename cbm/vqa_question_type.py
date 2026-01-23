import json
from collections import defaultdict
import torch

# 定义问题类型及其关键词
question_types = {
    "how_many_questions": ["how many", "how much", "how"],
    "is_questions": [
        "is the", "is this", "are the", "is there", "is it", "are those", "are these", "is someone",
        "is there", "is he", "was", "is that", "is one", "is a", "is she", "are this", "are this", "are both"
        "is someone", "are both", "is his", "is her", "are her", "are his"
    ],
    "all_any_question": ["are all", "are any", "are there any", "has any", "is everyone","any", "all"],
    "what_questions": [
        "what", "what is", "what are the"
    ],
    "what_kind_questions": [
        "what kind of", "what type of"
    ],
    "what_color_questions": [
        "what color is", "what is the color"
    ],
    "what_location_questions": [
        "what is on the", "what is in the"
    ],
    "what_specifics_questions": [
        "what time", "what sport", "what is the name", "what brand", "what number is"
    ],
    "does_do_questions": [
        "does the", "do", "do you", "did"
    ],
    "which_questions": ["which"],
    "why_questions": ["why"],
    "where_questions": ["where are", "where is", "where"],
    "who_questions": ["who is", "who are", "who is", "who had", "whose", "who"],
    "can_questions": ["can you", "could", "can people", "can this", "can"],
    "will_would_questions": ['would', "will"],
    "has_had_questions": ["has the", "have the", "has this", "has it", "has he", "has she"],
    "none_of_the_above_questions": ["none of the above"],
}

# 读取问题数据
questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
with open(questions_path, 'r') as f:
    questions = json.load(f)

# 初始化分类结果字典
classified_questions = defaultdict(list)
questions = questions['questions'][:10000]
# 遍历问题并分类
for i, item in enumerate(questions):
    
    question_text = item['question'].lower()  # 将问题转为小写以进行匹配
    classified = False
    
    for question_type, keywords in question_types.items():
        if any(keyword in question_text for keyword in keywords):
            # classified_questions[question_type].append((i, question_text))
            classified_questions[question_type].append(i)
            classified = True
            break
    
    if not classified:
        # classified_questions["none_of_the_above_questions"].append((i, question_text))
        classified_questions["none_of_the_above_questions"].append(i)

# 统计每一类中的样本个数
question_counts = {question_type: len(questions) for question_type, questions in classified_questions.items()}

# 输出结果
print("分类结果：")
for question_type, questions in classified_questions.items():
    print(f"{question_type}: {len(questions)}")

print("\n每类样本个数：")
for question_type, count in question_counts.items():
    print(f"{question_type}: {count}")


print(classified_questions["none_of_the_above_questions"])

torch.save(classified_questions, "./domain/classified_questions.pth")

# ["how many", ]
# ["is the", "is this", "are the", "is there", "is it", "is there", "are there any", "is he", "was", "is that"]

# ["what", "what is", "what are the"]
# ["what kind of", "what type of"]
# ["what color is", "what is the color"]
# ["what is on the", "what is in the"]
# ["what time", "what sport", "what is the name", "what brand", "what number is"]

# ["does the", "do", "do you"]

# ["which"]

# ["why"]

# ["where are"]

# ["who is"]

# ["can you", "could", ]

# ["none of the above"]


