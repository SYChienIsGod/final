1. compilation:
   make

2. training :
   ./rnnlm -train data/train.txt -valid data/valid -rnnlm model/model_v1 -hidden 40 -rand-seed 1 -debug 2 -bptt 3 -class 200

3. testing :
   ./rnnlm -rnnlm model/model_v1 -test data/test

3. rescoring :
   ./rnnlm -rnnlm model/model_v1 -test data/nbest_ex.txt -nbest -debug 0 > data/scores_ex.txt


-------------------- RNNLM model --------------------
1. model_v1 : train with train.txt, valid with valid, hidden 50, bptt 3, class 100  20150621 by HYTseng
              logP = -1417.764546, PPL 26.167639 on testing example test file provided by rnnlm

2. model_v2 : train with train.txt, valid with valid, hidden 100, bptt 4, class 300, bptt block 10, RS 2 20150622 by HYTseng
              logP = -1199.139929, PPL 15.817576 on testing example test file provided by rnnlm

3. model_v3 : the same, hidden 70, bptt 3, class 300, bptt-block 5, rand-seed 2, 20150623 by HYTseng
              logP = 968.834569, PPL = 9.307533

4. model_v4 : the same, hidden 90, bptt 5, class 330, bptt-block 5, rand-seed 3

-------------------- SRILM model --------------------
1. model_v1 : train with test.txt, order 3, -kndiscount -interpolate -gt3min 1 -gt4min 1

2. model_v2 : train with test.txt, order 5, the same

3. model_v3 : train with train, order 3, the same

-------------------- BLEND model --------------------
1. Rmodel_v2
2. Rmodel_v3*0.6, Smodel*0.4, score 7.57
3. Rmodel_v3*0.3, Smodel*0.7, score 7.60
