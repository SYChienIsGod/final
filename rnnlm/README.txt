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
              logP = -20136, PPL 280.48 on testing example test file provided by rnnlm

2. model_v2 : train with train.txt, valid with valid, hidden 100, bptt 4, class 300, bptt block 10, RS 2 20150622 by HYTseng
              logP = -20034, PPL 272.58 on testing example test file provided by rnnlm

3. model_v3 : the same, hidden 70, bptt 3, class 300, bptt-block 5, rand-seed 2, 20150623 by HYTseng
              logP = -19905, PPL = 262.58

4. model_v4 : the same, hidden 90, bptt 5, class 330, bptt-block 5, rand-seed 3
              logP = -21174, PPL = 375.09
              
5. model_v5 : train/valid, hidden 90. bptt 5, class 400, bptt-block 5, rand-seed 7
              logP = -21682, PPL = 391.95

6. model_v6 : train_debug/valid_debug, hidden 90, bptt 3, class 400, bptt-block 5, rand-seed 2
              logP = -21724, PPL = 397.42
-------------------- SRILM model --------------------
1. model_v1 : train with test.txt, order 3, -kndiscount -interpolate -gt3min 1 -gt4min 1
              logP = -20266, ppl = 290.852, ppl1 = 637.702
2. model_v2 : train with test.txt, order 5, the same
              logP = -20257, ppl = 290.179, ppl1 = 636.022
3. model_v3 : train with train, order 3, the same
              logP = -20557, ppl = 287.519, ppl1 = 620.411
4. model_v4 : train with timit_train, order 3, the same
              logP = -18275.6, ppl = 294.901, ppl1 = 717.072
5. model_v5 : train with train_debug, order 3, the same
              logP = -20845.7, ppl = 311, ppl1 = 680.882
-------------------- BLEND model --------------------
1. Rmodel_v2
2. Rmodel_v3*0.6, Smodel*0.4, score 7.57
3. Rmodel_v3*0.3, Smodel*0.7, score 7.60
4. phone_seq_v3(caffe_smooth), Rmodel_v3*0.6, Smodel_v3*0.4, score 7.71
5. phone_seq_v4(caffe + SSVM best 1), Rmodel_v3*0.6, Smodel_v3*0.4, score 6.67
6. phone_seq_v4(caffe + SSVM best 1), Rmodel_v3*0.5, Smodel_v5*0.5, score 6.68
7. blend with 5(0.6) + 6(0.4), score 6.674
