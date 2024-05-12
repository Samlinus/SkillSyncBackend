import heapq
from tqdm import tqdm
import pandas as pd


class SkillScore:
    def __init__(self, string, integer):
        self.skill_name = string
        self.skill_score = integer * 100

    def __lt__(self, other):
        return self.skill_score > other.skill_score


class Data:
    def __init__(self, df, graph, real_adj_mat, label_enc):
        self.df = df
        self.graph = graph
        self.real_adj_mat = real_adj_mat
        self.label_enc = label_enc


class KaggleModel:

    def __init__(self, pickle_data: Data):
        self.data = pickle_data

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
    def find_score(self, index: int, df: pd.DataFrame):
        similar_skills = []
        # df.iloc[index].iloc[1] -> Gives the skill set of each user
        for skill in tqdm(df.iloc[index, 0]):
            # Getting heap that contains of scores of each users..
            heap = self.get_similar_skills(skill, self.data.graph, self.data.real_adj_mat)
            # Considering 1 highest similar skill for each use's skill
            similar_skills.append(heapq.heappop(heap).skill_name)

        # Add the encodings of similar skills of each skills to the user vector..

        similar_skills_encodings = {self.data.label_enc[skill] for skill in similar_skills}
        encodings = df.iloc[index, 1]
        # update encoding - user skill encoding + similar skill encoding
        encodings.update(similar_skills_encodings)

        # Finding scores between each users after updating the skill vector user at index 'index'
        heap = []
        for j in range(0, 100):
            if index != j:
                # In the SkillScore class,
                # Skill_name = user name
                # skill_score = jaccard score
                heapq.heappush(heap,
                               SkillScore(
                                   df.index[j - 1], self.jaccard_similarity(df.iloc[j].iloc[1], encodings)
                               )
                               )

        return heap

    def find_similar_users(self, actual_user_index, k, df: pd.DataFrame):
        heap = self.find_score(actual_user_index, df)
        return self.find_top_k_users(k, heap)


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
        return sim_users, self.skills_of_users(sim_users)

    def generate_encodings(self, skills: {}) -> []:
        return {self.data.label_enc[skill] for skill in skills}

    def skills_of_users(self, users: []) -> []:
        return [list(self.user_data.loc[user]['skills']) for user in users]


class UserData:
    def __init__(self, name: str, skills: []):
        self.name = name
        self.skills = skills
