
nsml run -d movie_phase2 -a "--model=LSTM_Attention --normalizer=BasicNormalizer --tokenizer=TwitterTokenizer --dictionary=FastTextVectorizer --features=MovieActorFeaturesExtractor_AbnormalWordExtractor_ScoreExpressionExtractor --epochs=100 --sort_dataset --learning_rate=0.003 --min_length=5 --max_length=50 --embedding_size=256 --vocabulary_size=50000"
nsml run -d movie_phase2 -a "--model=LSTM_Attention --normalizer=BasicNormalizer --tokenizer=TwitterTokenizer --dictionary=FastTextVectorizer --features=MovieActorFeaturesExtractor_AbnormalWordExtractor_ScoreExpressionExtractor --epochs=100 --sort_dataset --lr_schedule --learning_rate=0.05 --lr_schedule --min_length=5 --max_length=50 --embedding_size=256 --vocabulary_size=50000"

python3 main.py --model=FastText --normalizer=BasicNormalizer --tokenizer=TwitterTokenizer --dictionary=FastTextVectorizer --epochs=30 --learning_rate=0.003 --min_length=5 --max_length=50 --embedding_size=300 --vocabulary_size=50000

naml run -d movie_phase2 -a "--model=FastText --normalizer=BasicNormalizer --tokenizer=TwitterTokenizer --dictionary=FastTextVectorizer --epochs=30 --learning_rate=0.003 --min_length=5 --max_length=50 --embedding_size=300 --vocabulary_size=50000"


