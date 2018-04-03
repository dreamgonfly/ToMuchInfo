
class LengthFeatureExtractor:
    """A dummy feature extractor that counts the number of tokens"""
    
    def __init__(self, config):
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
