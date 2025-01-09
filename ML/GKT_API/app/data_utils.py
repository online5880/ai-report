import numpy as np
import torch

def prepare_data(user_data, next_skills, input_data):
    """
    입력 데이터를 준비하는 함수.
    """
    try:
        features = np.array(user_data["skill_with_answer"].to_list(), dtype=np.int64)
        questions = np.array(user_data["skill"].to_list(), dtype=np.int64)

        next_skills_array = np.array(next_skills, dtype=np.int64)
        correct_list_array = np.array(input_data.correct_list, dtype=np.int64)

        features = np.concatenate([features, next_skills_array * 2 + correct_list_array])
        questions = np.concatenate([questions, next_skills_array])

        features_tensor = torch.from_numpy(features).unsqueeze(0)
        questions_tensor = torch.from_numpy(questions).unsqueeze(0)

        return features_tensor, questions_tensor
    except Exception as e:
        print(f"입력 데이터 준비 중 오류 발생: {e}")
        raise
