1. compilation:
   make

2. training :
   ./rnnlm -train data/train.txt -valid data/valid -rnnlm model/model_v1 -hidden 40 -rand-seed 1 -debug 2 -bptt 3 -class 200

3. testing :
   ./rnnlm -rnnlm model/model_v1 -test data/test

3. rescoring :
   ./rnnlm -rnnlm model/model_v1 -test data/nbest_ex.txt -nbest -debug 0 > data/scores_ex.txt


History:

1. model_v1 : train with train.txt, valid with valid, hidden 50, bptt 3, class 100  20150621 by HYTseng
              logP = -1417.764546, PPL 26.167639 on testing example test file provided by rnnlm
