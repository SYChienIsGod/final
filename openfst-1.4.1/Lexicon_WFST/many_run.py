import os
import errno
import sys
if __name__=='__main__':
    print sys.argv
    print len(sys.argv)
    filename = ''
    nbest = 1
    if len(sys.argv) == 1:
        print 'too few arguments'
        exit(-1)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        nbest = int(sys.argv[2])
    else:
        print 'too many arguments'
        exit(-1)
    print filename, nbest
    dir_name = 'temp'
    try:
        os.makedirs(dir_name)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
    fr = open(filename, 'r')
    line_num = 0
    for l in fr:
        fw_name = dir_name+'/'+filename+str(line_num)
        fw = open(fw_name, 'w')
        fw.write(l)
        line_num += 1
        fw.close()
        if nbest == 1:
            bashCommand = "./run.sh "+fw_name
        else:
            bashCommand = "./run.sh "+fw_name+ " " + str(nbest)
        os.system(bashCommand)
