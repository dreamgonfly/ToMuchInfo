# ToMuchInfo ([NAVER AI Hackathon](https://github.com/naver/ai-hackathon-2018))

**Author : 이상헌, 조용래, 박성남**

---
### Abstract
 `Preprocessing` - `Tokenize` - `Feature Extraction` - `Embedding` - `Model` - `Ensemble(optional)`
 
### Preprocessing
- `normalizers.py` : correct bad words and typo
- `LSUV.py`, `ironyer.py` : applied LSUV init (https://arxiv.org/pdf/1511.06422.pdf)

### Tokenize
- `DummyTokenizer` : dummy tokenizer that splits a sentence by space
- `JamoTokenizer` : split text into jamos
- `JamoMaskedTokenizer` : split text into jamos and mask movie names and actor names
- `TwitterTokenizer` : tokenize text using konlpy's Twitter module
- `SoyNLPTokenizer` : tokenize text using SoyNLP's MaxScoresTokenizer

### Feature Extraction (`feature_extractors.py`)
- `LengthFeatureExtractor` : token의 길이
- `ImportantWordFeaturesExtractor` : 부정적 단어, 욕설 단어, 반전 단어의 수
- `MovieActorFeaturesExtractor` : 자주 언급된 배우/영화를 찾고 이를 one-hot encoding
- `AbnormalWordExtractor` : 직접 데이터를 보며 유의미할 것 같은 단어들 one-hot encoding
- `SleepnessExtractor` : 졸리다는 내용의 표현 수

### Embedding (`dictionaries.py`)
- `RandomDictionary` : 단순히 word를 index화 시켜서 return
- `FastTextDictionary` : pretrained FastText embedding을 불러와 embedding
- `FastTextVectorizer` : train set으로 FastText 학습시키고 embedding
- `Word2VecVectorizer` : train set으로 Word2Vec 학습시키고 embedding
- `TfidfVectorizer` : train set으로 sklearn을 사용해 tf-idf vectorize

### Model
- `VDCNN` : [Very Deep Convolutional Networks
for Text Classification](https://arxiv.org/pdf/1606.01781.pdf)
- `WordCNN` : [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1606.01781.pdf)
- `BiLSTM` : [Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](https://arxiv.org/pdf/1611.06639.pdf)
- `CNNTextInception` : [Merging Recurrence and Inception-Like
Convolution for Sentiment Analysis](https://cs224d.stanford.edu/reports/akuefler.pdf)
- `DCNN-LSTM` : 저희 팀이 만들었습니다.
- `LSTM_Attention` : [Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification](http://www.aclweb.org/anthology/P16-2034)
- `RCNN` : [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- `TDSM` : [Character-Based Text Classification using Top Down Semantic Model for Sentence Representation](https://arxiv.org/pdf/1705.10586.pdf)

### Ensemble
- Average : 한 epoch당 여러개의 모델을 동시에 돌리면서, validation loss가 더이상 떨어지지 않으면 저장. 성능이 잘나오는 모델만 모아서 평균을 냄
- XGBRegressor : 각각의 모델당 best epoch을 찾은 후, 한번에 다 돌리고 그 결과값들로 xgboost를 써서 예측
