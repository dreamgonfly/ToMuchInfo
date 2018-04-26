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
        #### 상헌
        normalized_text = common_errors(normalized_text)
        normalized_text = common_mispel(normalized_text)
        normalized_text = common_abbr(normalized_text)
        normalized_text = common_eng(normalized_text)
        normalized_text = common_propnoun(normalized_text)

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
ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}ㅂ|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔|x발|x빨|X발|X빨')
    for sibal in r.findall(raw_text):
        raw_text = raw_text.replace(sibal, "시발")
    return raw_text


def bad_words_exchanger(raw_text):
    bad_words = {'쓰레기': {'쓰레기', '쓰래기', 'ㅆㄹㄱ', '쓰렉', '쓰뤠기', '레기', '쓰렉이', '쓰렣귀','ㅅㄹㄱ'},
                 '병신': {'병신', '븅신', '빙신', 'ㅂㅅ','병`신',"병'신"},
                 '존나': {'존나', '졸라', '조낸', '존내', 'ㅈㄴ', '존니', '좆나', '좆도', '좃도', '좃나', '죤냐'},
                 '뻐큐': {'뻐큐', '뻑큐', '凸'},
                 '열라': {'연나','욘나','욜라'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text


def common_errors(raw_text):
    bad_words = {'없는': {'업는'},
                 '없다': {'없다'},
                 '았다': {'앗다'},
                 '의리': {'으리', 'ㅇㄹ'},
                 '재미': {'제미'},
                 '잼': {'젬'},
                 '밌': {'밋'},
                 '에요': {'예여','에여'},
                 '마이너스': {'마이나스'},
                 '럽게': {'럽개'},
                 '지지리': {'지질이'},
                 '코미디': {'코메디'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text

def common_mispel(raw_text):
    bad_words = {'없는': {'업는'},
                 '없다': {'업다'},
                 '았다': {'앗다'},
                 '의리': {'으리', 'ㅇㄹ'},
                 '그렇게': {'글케'},
                 '정말': {'증말'},
                 '그냥': {'근양'},
                 '뒤질': {'듸질'},
                 '는데': {'는디','는듸'},
                 '뒤지는': {'디지는','듸지는'},
                 '없음': {'업슴','업음','없슴','업씀','없씀'},
                 '있음': {'잇슴','잇음','있슴','이씀','있씀','있슴'},
                 '굿': {'굳','구웃','구욷','구[우]+웃','긋~','긋!'}, # 여기서도 구우우우우웃 어케잡지
                 '있다': {'이따','있따',},
                 '되게': {'디게'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text

def common_abbr(raw_text):
    bad_words = {'콩까지마': {'ㅋㄲㅈㅁ'},
                 '고고': {'ㄱㄱ'},
                 '재미있': {'재밌','잼있'},
                 '재미없': {'잼없','재미 없'},
                 '재미있다': {'잼써','잼따'},
                 '있어': {'있써'}, # 3번째 때문에
                 '없어': {'없써'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text

def common_eng(raw_text):
    bad_words = {'요': {'yo'},
                '굿바이':{'good bye','goodbye'},
                 '굿': {'go[o]+d',}, # gooooooooooooooood 어케잡음
                 '디비디':{'dvd'},
                 '판타스틱':{'fantastic'},
                 '오에스티':{'ost'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text

def common_eng(raw_text):
    bad_words = {'퍼센트': {'%'},
                '하트':{'♥','♡'},

                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text

def common_propnoun(raw_text):
    bad_words = {'씨지비': {'cgv','시지비','씨쥐비'},
                '하트':{'♥','♡'},

                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(raw_text)
        for token in tokens:
            raw_text = raw_text.replace(token, bad_word)

    return raw_text
