
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
            raw_text = raw_text.replace("'", '').replace('"', '')
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

        """Normalize text in a dumb way

        Args:
            raw_text: A string

        Returns:
            A string
        """
        normalized_text = delete_quote(raw_text)
        normalized_text = sibalizer(raw_text)
        return normalized_text
