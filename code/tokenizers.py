
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
