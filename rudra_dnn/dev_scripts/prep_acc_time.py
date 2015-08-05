import sys
if __name__ == "__main__":
    #print "USAGE: python pre_acc_time.py phase1.dat phase2.dat phase3.dat ..."
    curFileNum = 0
    curFile = ""
    curEpoch = 0
    curTrainErr = 0.0
    curTestErr = 0.0
    curTime = 0.0
    curEpochGenesis = 0
    curTimeGenesis = 0.0
    for arg in sys.argv[1:]:
        curFile = arg
        curFileNum+=1
        curEpochGenesis = curEpoch
        curTimeGenesis   = curTime
        with open(arg) as f:
            content = f.readlines();
            for line in content:
                infoL = filter(None, line.strip().split("\t"))
                curTrainErr = float(infoL[1])
                curTestErr  = float(infoL[2])
                curEpoch = curEpochGenesis + int(infoL[0])
                curTime  = curTimeGenesis + float(infoL[3])
                print str(curEpoch)+"\t"+str(curTrainErr)+"\t"+str(curTestErr)+"\t"+str(curTime)

                    
  
