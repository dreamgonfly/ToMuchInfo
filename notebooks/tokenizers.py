from jamo import h2j, j2hcj
from konlpy.tag import Twitter
import re

class DummyTokenizer:
    """A dummy tokenizer that splits a sentence by space"""

    def __init__(self, config):
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
    
    
if __name__ == '__main__':

    tokenizer = JamoMaskedTokenizer(None)
    assert tokenizer.tokenize("ac01431291의 출연만으로도 충분히 mv00069433.") == "🐱ㅇㅢ ㅊㅜㄹㅇㅕㄴㅁㅏㄴㅇㅡㄹㅗㄷㅗ ㅊㅜㅇㅂㅜㄴㅎㅣ 🐶."
