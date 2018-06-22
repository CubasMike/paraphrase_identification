# Original
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1

# Ablation study
# 1
python he_approach.py --embeddings glove paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 325 --fnum-gb 20 --inf-ga --no-abs-a1
python he_approach.py --embeddings glove POSword2vec --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 500 --fnum-gb 20 --inf-ga --no-abs-a1

# 2
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1 --no-groupb
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max --pool-gb max --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 1 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1

# 3
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-cos-a1 --no-cos-a2 --no-euc-a1 --no-euc-a2

# 4
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1 --no-algo1
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1 --no-algo2 --no-groupb
python he_approach.py --embeddings glove POSword2vec paragram25 --verbose 2 --epochs 30 --depthwise-conv2d --pool-ga max min mean --pool-gb max min --ngrams 3 --fnum-ga 525 --fnum-gb 20 --inf-ga --no-abs-a1 --no-algo1 --no-algo2
