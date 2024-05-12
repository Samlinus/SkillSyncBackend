import heapq
import pickle

import pandas as pd
from tqdm import tqdm

from Modules.classfiles import Data, SkillScore

with open('F:\\RecommendationSystem\\RecommendationApp\\RecommendationApp\\data.pickle', 'rb') as f:
    data = pickle.load(f)

with open('F:\\RecommendationSystem\\RecommendationApp\\RecommendationApp\\userdata.pickle', 'rb') as f:
    user_data = pickle.load(f)


class Model:
    def __init__(self, pickle_data: Data, user_data: pd.DataFrame):
        self.data = pickle_data
        self.user_data = user_data

    @staticmethod
    def jaccard_similarity(a: set[int], b: set[int]):
        return len(a.intersection(b)) / len(a.union(b))

    def get_similar_skills(self, skill: str, graph, adjacency_matrix: []):
        heap = []
        skill_index = self.data.label_enc[skill]
        # list(graph.nodes()) - contains the list of nodes i.e, skills
        labels_list = list(graph.nodes())
        for index, vector in enumerate(tqdm(adjacency_matrix)):
            heapq.heappush(heap, SkillScore(
                labels_list[index],
                # Sim = (A U B)/ (A n B)
                self.jaccard_similarity(adjacency_matrix[skill_index], vector)
            ))

        return heap

    def find_top_k_users(self, k: int, heap: []):
        return [heapq.heappop(heap).skill_name for _ in range(k)]

    # Compares the scores between the skills of each users..
    def find_score(self, skills: {}):
        similar_skills = []
        # Skills contains the skills of the user
        for skill in tqdm(skills):
            # Getting heap that contains of scores of each users..
            heap = self.get_similar_skills(skill, self.data.graph, self.data.real_adj_mat)
            # Considering 1 highest similar skill for each use's skill
            similar_skills.append(heapq.heappop(heap).skill_name)

        # Add the encodings of similar skills of each skills to the user vector..
        similar_skills_encodings = {self.data.label_enc[skill] for skill in similar_skills}
        encodings = self.generate_encodings(skills)
        # update encoding - user skill encoding + similar skill encoding
        encodings.update(similar_skills_encodings)

        # Finding scores between each users after updating the skill vector user at index 'index'
        heap = []
        for j in range(0, 100):
            # In the SkillScore class,
            # Skill_name = user name
            # skill_score = jaccard score
            heapq.heappush(heap,
                           SkillScore(
                               self.user_data.index[j - 1],
                               self.jaccard_similarity(self.user_data.iloc[j].iloc[1], encodings)
                           )
                           )

        return heap

    def find_similar_users(self, skills: {}, k):
        heap = self.find_score(skills)
        # Construct heap to find to rank the use profile scores
        sim_users = self.find_top_k_users(k, heap)
        return sim_users, self.skill_of_users(sim_users)

    def generate_encodings(self, skills: {}) -> []:
        return {self.data.label_enc[skill] for skill in skills}

    def skill_of_users(self, users: []) -> []:
        return [list(self.user_data.loc[user]['skills']) for user in users]


model = Model(data, user_data)

from joblib import dump

dump(model, 'D:\\DjangoProject\\finalproject\\Modules\\model.joblib')
print('Successfully saved model')
