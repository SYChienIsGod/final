1. Download trained model from MediaG
      >> scp -P 5566 mlds@140.112.20.167:caffe_model/515_PReLU_7413.tar .
      passwore: 5566
      (There are some trained model in folder, download the model you like)      
      Model name example
      EX. 515_PReLU_5_741 
      -> [data input(5+1+5)] ___ [ReLU or PReLU] ___ Hidden Layer count ___ Val. Accuracy
      
	├── basenet.prototxt 			(specify training model)
	├── basenet.sh       			(no use)
	├── basenet.solver			(specify training parameter)
	├── log.txt				(training log)
	├── snapshot_iter_5544000.caffemodel	(trained model)
	├── snapshot_iter_5544000.solverstate	(trained model)
	└── train.sh				(training scrip)
      
2. To plot accuracy curve (Remember to modify model path)
      >> PlotLog.sh

3. To resume training & see val. accuracy
      >> resume_train.sh

4. To predict result (you can do model blending here)
      >> python predict_caffe.py 

5. If you can NOT run code on MediaG (GPU server) try this
      >> source source.sh

6. If you still have problem, go to my home @MediaG to see what I did
      ssh XXX@140.112.20.167 -p 5566
      "/home/baconx2/MLDS/final/final"
