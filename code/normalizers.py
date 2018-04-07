
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
