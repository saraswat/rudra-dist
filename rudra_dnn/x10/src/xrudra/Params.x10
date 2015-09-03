package xrudra;

import xrudra.util.Logger;
import xrudra.util.Timer;

public struct Params(
                numEpochs:Long,
                mbPerPlace:Long,
                mbSize:Long,
                trainingNum:Long
                
){
        public static Default = Params(2l,3l,5l,100l);
}
