from jamo import h2j, j2hcj
import re
from konlpy.tag import Twitter


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

class TwitterTokenizer:
    """Noun, Adjective, Verb만 output 내는 tokenizer"""

    def __init__(self, config):
        self.tw = Twitter()
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
