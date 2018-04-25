from jamo import h2j, j2hcj
import re
from konlpy.tag import Twitter
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from collections import Counter


class DummyTokenizer:
    """A dummy tokenizer that splits a sentence by space"""

    def __init__(self, config):
        pass

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text):
        """Tokenize raw text

        Args:
            raw_text: A string of raw text. For example : "무궁화 꽃이 피었습니다."

        Returns:
            A list of tokens. For example:

            ['무궁화', '꽃이', '피었습니다.']
        """

        return raw_text.split()


class JamoTokenizer:
    """Split text into jamo's"""

    def __init__(self, config):
        self.movie_actor = re.compile(r"mv[0-9]*|ac[0-9]*|.")

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text):
        """

        :param raw_text: "아이들이 보면 좋을 mv00217576~"
        :return: ['ㅇ', 'ㅏ', 'ㅇ', 'ㅣ', 'ㄷ', 'ㅡ', 'ㄹ', 'ㅇ', 'ㅣ', ' ', 'ㅂ', 'ㅗ', 'ㅁ', 'ㅕ', 'ㄴ', ' ', 'ㅈ', 'ㅗ', 'ㅎ', 'ㅇ', 'ㅡ', 'ㄹ', ' ', 'mv00217576', '~']

        """
        jamo_text = j2hcj(h2j(raw_text))
        return self.movie_actor.findall(jamo_text)


class JamoMaskedTokenizer:
    """Split text into jamo's and mask movie names and actor names"""

    def __init__(self, config):
        self.mv = re.compile(r'mv[0-9]{2,10}')
        self.ac = re.compile(r'ac[0-9]{2,10}')

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text):
        """Tokenize text into jamo with actors and movies masked

        Args:
            raw_text: A string. For example: "ac01431291의 출연만으로도 충분히 mv00069433."

        Returns:
            A tokenized and masked string. For example:

            "🐱ㅇㅢ ㅊㅜㄹㅇㅕㄴㅁㅏㄴㅇㅡㄹㅗㄷㅗ ㅊㅜㅇㅂㅜㄴㅎㅣ 🐶."
        """
        jamo_text = j2hcj(h2j(raw_text))
        mv_replaced = self.mv.sub('🐶', jamo_text)
        ac_replaced = self.ac.sub('🐱', mv_replaced)
        return ac_replaced

class TwitterTokenizer:
    """Split text to twitter based tokens"""

    def __init__(self, config):
        self.twitter = Twitter()
        self.mv = re.compile(r'mv[0-9]{2,10}')
        self.ac = re.compile(r'ac[0-9]{2,10}')

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text, stem=False):
        """
        Args:
            raw_text: "무궁화 꽃이 피었습니다."
        Returns:
            먼저 영화id와 배우id를 masking
            A list of (token, pos) : [("무궁화","Noun"), ("꽃","Noun")...]
        """
        mv_replaced = self.mv.sub('🐶', raw_text)
        ac_replaced = self.ac.sub('🐱', mv_replaced)
        tokenized_text = self.twitter.pos(ac_replaced, stem=stem)
        idx_mv = []
        idx_ac = []
        for i, (token, pos) in enumerate(tokenized_text):
            if token=='\uf436':
                idx_mv.append(i)
            elif token=='\uf431':
                idx_ac.append(i)

        for i in idx_mv:
            tokenized_text[i] = ('🐶', 'Movie')
        for i in idx_ac:
            tokenized_text[i] = ('🐱', 'Actor')

        return tokenized_text

class TwitterTokenizer_SH:
    """Noun, Adjective, Verb만 output 내는 tokenizer"""

    def __init__(self, config):
        self.tw = Twitter()
        pass

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text):
        """Noun, Verb, Adjective output 내는 tokenizer
        Args:
            raw_text: A string of raw text. For example : "무궁화 꽃이 피었습니다."
        Returns:
            A list of tokens. For example:
            ['무궁화', '꽃이', '피었습니다.']
        """
        poses = self.tw.pos(raw_text)
        output = []
        for word, pos in poses:
            if(pos == "Noun" or pos == "Verb" or pos == "Adjective"):
                output.append(word+'_'+pos)
        return output


class MultiTokenizer:
    def __init__(self, config):
        self.tokenizers = [JamoTokenizer, TwitterTokenizer_SH, ]
        self.config = config

    def fit(self, data):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def tokenize(self, raw_text):
        output = []

        for tokenizer in self.tokenizers:
            output += (tokenizer(self.config).tokenize(raw_text))
        return output


class SoyNLPTokenizer:
    def __init__(self, config):
        self.tokenizer = None
        self.scores = list()
        self.word_extractor = WordExtractor(min_count=100,
                                            min_cohesion_forward=0.05,
                                            min_right_branching_entropy=0.0)
        self.popular_ids = list()
        self.re_movie_actor = re.compile("mv[0-9]*|ac[0-9]*")

    def fit(self, data):
        # reviews 뽑기
        reviews = [review for review, label in data]

        # 30번 이상 언급된 영화, 배우이름 뽑는 과정
        ids_count = []
        for review in reviews:
            movie_actor_token = self.re_movie_actor.findall(review)
            if movie_actor_token:
                ids_count += movie_actor_token
        ids_count = Counter(ids_count)
        ids_count = {id:freq for id,freq in ids_count.items() if freq>30}
        self.popular_ids = set(ids_count.keys())

        # 각각의 review에 대해, 유명한 영화/배우이면 그 자체로 토큰을 넣고, 아니면 개/고양이 모양으로 바꿈
        normalized_reviews = []
        for review in reviews:
            movie_actor_token = self.re_movie_actor.findall(review)
            for token in movie_actor_token:
                if token in self.popular_ids:
                    pass
                elif token[:2] == 'mv':
                    review = review.replace(token, '🐶')
                elif token[:2] == 'ac':
                    review = review.replace(token, '🐱')
                else:
                    print("뭔가잘못되었어!!!!!!!!!!!!!!!!")
            normalized_reviews.append(review)

        # tokenizer 학습
        self.word_extractor.train(normalized_reviews)
        scores = self.word_extractor.extract()
        scores = [(word, (score.cohesion_backward + score.cohesion_backward) * \
                   (score.left_branching_entropy+score.right_branching_entropy))
                  for word, score in scores.items()]
        scores = {word:score for word, score in scores if score>0}
        for popular_id in self.popular_ids:
            scores.update({popular_id:20})
        self.scores = scores
        self.tokenizer = MaxScoreTokenizer(scores=self.scores)

    def state_dict(self):

        return {'scores': self.scores,
                'popular_ids': self.popular_ids}

    def load_state_dict(self, state_dict):
        self.scores = state_dict['scores']
        self.popular_ids = state_dict['popular_ids']
        self.tokenizer = MaxScoreTokenizer(scores=self.scores)

    def tokenize(self, raw_text):

        movie_actor_token = self.re_movie_actor.findall(raw_text)
        for token in movie_actor_token:
            if token in self.popular_ids:
                pass
            elif token[:2] == 'mv':
                raw_text = raw_text.replace(token, '🐶')
            elif token[:2] == 'ac':
                raw_text = raw_text.replace(token, '🐱')
            else:
                print("뭔가잘못되었어!!!!!!!!!!!!!!!!")

        return self.tokenizer.tokenize(raw_text)


class JamoUnpopularMaskedTokenizer:
    def __init__(self, config):
        self.popular_ids = list()
        self.re_movie_actor = re.compile(r"mv[0-9]+|ac[0-9]+")
        self.re_all = re.compile(r"mv[0-9]+|ac[0-9]+|&#\d{1,6};|.")

    def fit(self, data):
        # reviews 뽑기
        reviews = (review for review, label in data)

        # 30번 이상 언급된 영화, 배우이름 뽑는 과정
        movie_actor_reviews = [self.re_movie_actor.findall(review) for review in reviews]
        movie_actor_whole_list = sum(movie_actor_reviews, [])
        counter = Counter(movie_actor_whole_list)
        self.popular_ids = {name for name, freq in counter.items() if freq > 30}

    def tokenize(self, raw_text):
        jamo_text = j2hcj(h2j(raw_text))
        movie_actor_tokens = self.re_movie_actor.findall(jamo_text)
        for token in movie_actor_tokens:
            if token in self.popular_ids:
                pass
            elif token[:2] == 'mv':
                jamo_text = jamo_text.replace(token, '🐶')
            elif token[:2] == 'ac':
                jamo_text = jamo_text.replace(token, '🐱')
            else:
                raise NotImplementedError("뭔가 잘못되었어!!!")
        tokenized = self.re_all.findall(jamo_text)
        return tokenized

    def state_dict(self):
        return {'popular_ids': self.popular_ids}

    def load_state_dict(self, state_dict):
        self.popular_ids = state_dict['popular_ids']


if __name__ == '__main__':

    tokenizer = SoyNLPTokenizer()
    assert tokenizer.tokenize("ac01431291의 출연만으로도 충분히 mv00069433.") == "🐱ㅇㅢ ㅊㅜㄹㅇㅕㄴㅁㅏㄴㅇㅡㄹㅗㄷㅗ ㅊㅜㅇㅂㅜㄴㅎㅣ 🐶."
