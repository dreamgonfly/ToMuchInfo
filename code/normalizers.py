import re


class DummyNormalizer:
    """A dummy normalizer which does nothing"""

    def __init__(self, config):
        pass

    def normalize(self, raw_text):
        """Normalize text in a dumb way

        Args:
            raw_text: A string

        Returns:
            A string
        """

        normalized_text = raw_text
        return normalized_text


class BasicNormalizer:
    """A basic normalizer"""

    def __init__(self, config):
        pass

    def normalize(self, raw_text):
        def delete_quote(raw_text):
            raw_text = raw_text.replace("'", '').replace('"', '').strip()
            if raw_text.find("10자") > -1:
                raw_text = raw_text[:raw_text.find("10자")]
            return raw_text

        """Normalize text in a dumb way

        Args:
            raw_text: A string

        Returns:
            A string
        """

        normalized_text = delete_quote(raw_text)
        normalized_text = sibalizer(normalized_text)
        return normalized_text


class AdvancedNormalizer:
    """Unite bad words and translate important english to korean"""

    def __init__(self, config):
        pass

    def normalize(self, raw_text):
        normalized_text = raw_text.lower()  # 모든 영어 단어를 lower case로 바꿈
        normalized_text = deleate_quote(normalized_text)  # 10자 이후 삭제, 따옴표 삭제
        normalized_text = sibalizer(normalized_text)  # 시발 최적화
        normalized_text = bad_words_exchanger(normalized_text)  # 기타 다른 욕설 각각의 유사어 통일

        return normalized_text


def deleate_quote(raw_text):
    raw_text = raw_text.replace("'", '').replace('"', '').strip()
    if raw_text.find("10자") > -1:
        raw_text = raw_text[:raw_text.find("10자")]
    return raw_text


def sibalizer(raw_text):
    r = re.compile('씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빨|\
씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}벌|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}뻘|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}펄|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|신[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}방|\
ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}ㅂ|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔')
    for sibal in r.findall(raw_text):
        raw_text = raw_text.replace(sibal, "시발")
    return raw_text


def bad_words_exchanger(raw_text):
    bad_words = {'쓰레기': {'쓰레기', '쓰래기', 'ㅆㄹㄱ', '쓰렉', '쓰뤠기', '레기', '쓰렉이'},
                 '병신': {'병신', '븅신', '빙신', 'ㅂㅅ'},
                 '존나': {'존나', '졸라', '조낸', '존내', 'ㅈㄴ', '존니', '좆나', '좆도', '좃도', '좃나'},
                 '뻐큐': {'뻐큐', '뻑큐', '凸'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text
