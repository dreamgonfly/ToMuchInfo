import re
from collections import defaultdict, Counter
import numpy as np

class LengthFeatureExtractor:
    """A dummy feature extractor that counts the number of tokens"""
    
    def __init__(self, config):
        self.n = 1

    def fit(self, data):
        pass
    
    def extract_feature(self, raw_text, tokenized_text):
        """Count the number of tokens
        
        Args:
            raw_text: A string of raw text. For example: "무궁화 꽃이 피었습니다."
            tokenized_text: A list of tokens from raw text. For example: ['무궁화', '꽃이', '피었습니다.']
            
        Returns:
            A tuple that represents the counts of tokens.
        """
        
        counts = len(tokenized_text)
        return counts,  # make it a tuple

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

class BasicFeaturesExtractor:
    """
    Extracts basic features of raw text
    """
    def __init__(self, config):
        self.n = 2
    
    def fit(self, data):
        """
        Args:
            data : (raw_text, score)로 이루어진 list
        Returns:
            raise NotImplementedError if not implemented
        """
        pass
    
    def extract_feature(self, raw_text, tokenized_text):
        """
        Returns:
            [0] raw text의 길이
            [1] token의 수
        """
        return len(raw_text), len(tokenized_text)

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

class ImportantWordFeaturesExtractor:
    """
    Extracts negative words, bad words, reverse words, specific words('ㅋ','ㅜ') 
    """
    def __init__(self, config):
        self.re_badwods = re.cxompile("시[0-9]*발")
        self.n = 7
    
    def fit(self, data):
        """
        Args:
            data : (raw_text, score)로 이루어진 list
        Returns:
            raise NotImplementedError if not implemented
        """
        pass
    
    def extract_feature(self, raw_text, tokenized_text):
        """
        Count the number of bad words
        Returns:
            [0] 욕설 단어의 수
            [1] 부정적 단어의 수
            [2] 반전적 단어의 수
            [3] 'ㅋ' 단어의 수
            [4] 'ㅎ' 단어의 수
            [5] 'ㅜ' 단어의 수
            [6] 'ㅡ' 단어의 수
        """
        tokenized_words = set(tokenized_text)
        result = [0]*7
        negative_words = {'안', '못', '안봐', '안보', '안봤', '못보', '못잤', '못된', '안들', '안가', '안자', '안해'}
        reverse_words = {'지만', '러나', '근데', '허나'}
        
        result[0] = len(self.re_badwords.findall(raw_text))
        result[1] = len(negative_words & tokenized_words)
        result[2] = len(reverse_words & tokenized_words)
        result[3] = raw_text.count('ㅋ')
        result[4] = raw_text.count('ㅎ')
        result[5] = raw_text.count('ㅜ')
        result[6] = raw_text.count('ㅡ')
        
        return tuple(result)
    
    def state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        pass
    

class MovieActorFeaturesExtractor:
    """
    Extracts statistics of movie and actor if mentioned
    """
    def __init__(self, config):
        self.re_movie = re.compile("mv[0-9]*")
        self.re_actor = re.compile("ac[0-9]*")
        self.movies_dict = None
        self.actors_dict = None
        self.global_stat = None
        self.n = 4
    
    def fit(self, data, threshold=20):
        """
        Extract global features of movies and actors
        Args:
            data : (raw_text, score)로 이루어진 list
            threshold : overfitting을 방지하기 위해 threshold이상 언급된 영화, 배우만 반영
        """
        movies_dict = defaultdict(lambda : list())
        actors_dict = defaultdict(lambda : list())
        for (comment, score) in data:
            for m_id in self.re_movie.findall(comment):
                movies_dict[m_id].append(score)
            for a_id in self.re_actor.findall(comment):
                actors_dict[a_id].append(score)
        movies_dict = {movie:l for movie,l in movies_dict.items() if len(l)>threshold}
        actors_dict = {actor:l for actor,l in actors_dict.items() if len(l)>threshold}
        self.movies_dict = movies_dict
        self.actors_dict = actors_dict
        self.global_stat = (np.mean([x[1] for x in data]), np.std([x[1] for x in data]))
                            
    def extract_feature(self, raw_text, tokenized_text):
        """
        Returns:
            [0] 언급된 영화의 평균 평점. 없을 시 전체 영화의 평균평점, 두 개 이상 시 평균
            [1] 언급된 영화 평점의 표준편차. 없을 시 전체 영화 평점의 표준편차, 두 개 이상 시 평균
            [2] 언급된 배우의 평균 평점. 없을 시 전체 배우의 평균평점, 두 개 이상 시 평균
            [3] 언급된 배우 평점의 표준편차. 없을 시 전체 배우 평점의 표준편차, 두 개 이상 시 평균
        """
        movies_dict = self.movies_dict
        actors_dict = self.actors_dict
        
        movie_scores = []
        for m_id in self.re_movie.findall(raw_text):
            if m_id in movies_dict:
                movie_scores.append((np.mean(movies_dict[m_id]), np.std(movies_dict[m_id])))
        
        actor_scores = []
        for a_id in self.re_actor.findall(raw_text):
            if a_id in actors_dict:
                actor_scores.append((np.mean(actors_dict[a_id]), np.std(actors_dict[a_id])))
        result = [self.global_stat[0], self.global_stat[1]]*2
        
        if len(movie_scores)!=0:
            result[0] = np.mean([x[0] for x in movie_scores])
            result[1] = np.mean([x[1] for x in movie_scores])
        if len(actor_scores)!=0:
            result[2] = np.mean([x[0] for x in actor_scores])
            result[3] = np.mean([x[1] for x in actor_scores])
        
        return tuple(result)
    
    def state_dict(self):
        params = {'movies_dict': self.movies_dict,
                  'actors_dict': self.actors_dict,
                  'global_stat': self.global_stat}

        return params

    def load_state_dict(self, state_dict):

        self.movies_dict = state_dict['movies_dict']
        self.actors_dict = state_dict['actors_dict']
        self.global_stat = state_dict['global_stat']
