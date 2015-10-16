package rudra;

import rudra.util.Logger;
import rudra.util.Timer;

public struct Params(
                numEpochs:Long,
                mbPerPlace:Long,
                mbSize:Long,
                trainingNum:Long
                
){
        public static Default = Params(2l,3l,5l,100l);
}
