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
