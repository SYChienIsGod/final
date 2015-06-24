# Final
## Running Caffe

To run Caffe, execute `caffedata/basenet.sh`. It will first use `write_feat.py` to write the necessary data.

The data should be in a folder `..\data\` where you have extracted the files from the Kaggle website.

The script `write_feat.py` can be edited to change the size of the feature vector (i.e. features of how many segments before and after the label giving segment are incorporated) and which features are used.

## Generating FBANK features

To generate Filter Bank features (FBANK) ourselves, Python Speech Features (http://python-speech-features.readthedocs.org/) are used.
Call `python timitwav.py` and the process will be triggered. You can change the parameters like the number of filter banks that is used and how many temporal derivatives are used.
Typically, 40 filter banks and the frame energy together with their first and second temporal derivative are reported in the literatue. This gives 3*41=123 dimensions. 
`write_feat.py` is then used to assemble the data for caffe (the respective section is currently commented out).


## Generating SSVM Input Files

Call `python write_ssvm.py` to generate structured SVM input. Note that a folder named 'ssvmdata' needs to be created beforehand. The script furthermore accesses several numpy arrays that are created when `write_feat.py` is called (you can generate them easily by calling the script after commenting out the LevelDB writing stuff).
After this is done, switch to the svm_struct folder and edit `svm_struct_api.c`. The only thing you need to edit here is `N_STATES` and `N_FEATURES` which are both 48 at the moment (until we switch to 39 phoneme prediction). N Best is not yet implemented. Afterwards, run `nn_svm.sh c` where you can replace `c` with a cost factor, roughly ranging from 0.1 to 1000 where higher gives better results but requires more training time.