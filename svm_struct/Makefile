# Makefile for empty SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC = gcc
LD = gcc

CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall -std=c99
LDFLAGS =  $(SFLAGS) -O3 -lm -Wall
LIBS = -lm
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

all: svm_empty_learn svm_empty_classify

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump 

#-----------------------#
#----   SVM-light   ----#
#-----------------------#
svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_light_clean: 
	cd svm_light; make clean

#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_noexe: 
	cd svm_struct; make svm_struct_noexe

svm_struct_clean: 
	cd svm_struct; make clean


#-------------------------#
#----  SVM empty API  ----#
#-------------------------#

svm_empty_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o 
	$(LD) $(LDFLAGS) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o -o svm_empty_classify $(LIBS)

svm_empty_learn: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct_learn_custom.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o
	$(LD) $(LDFLAGS) svm_struct/svm_struct_learn.o svm_struct_learn_custom.o svm_struct_api.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o -o svm_empty_learn $(LIBS)


svm_struct_api.o: svm_struct_api.c svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_api.c -o svm_struct_api.o $(LIBS)

svm_struct_learn_custom.o: svm_struct_learn_custom.c svm_struct_api.h svm_light/svm_common.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_learn_custom.c -o svm_struct_learn_custom.o

run:
	wget https://www.dropbox.com/s/0b0f8lyzmdihugr/48_39.map?dl=0
	wget https://www.dropbox.com/s/pauz0wckzwpm312/48_idx_chr.map_b?dl=0
	wget https://www.dropbox.com/s/vsr19fnolqrxca7/fbank_nn_svm_test.save?dl=0
	wget https://www.dropbox.com/s/53qom98k7e6vdy6/fbank_nn_svm_training.save?dl=0
	wget https://www.dropbox.com/s/y1iuf1rglf30mnc/fbank_svm_testing.sents?dl=0
	./svm_empty_learn -c 1000 fbank_nn_svm_training.save fbank_nn_svm_c1000.model
	./svm_empty_classify fbank_nn_svm_test.save fbank_nn_svm_c1000.model fbank_nn_svm_test1000.output
	python post_process_svm_output.py