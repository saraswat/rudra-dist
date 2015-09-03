package xrudra;

import x10.util.concurrent.AtomicBoolean;
import xrudra.util.SwapBuffer;
import xrudra.util.BBuffer;

import xrudra.util.Logger;
import xrudra.util.Monitor;
import xrudra.util.Unit;

class ImmedReconciler(size:Long, maxMB: UInt, learner:ImmedLearner, 
                      reducer:AtLeastRAllReducer, logger:Logger) {
    var timeStamp:UInt = 0un; // incremented each time an all reduce is done

    def run(fromLearner:SwapBuffer[TimedGradient], done:AtomicBoolean) {
        logger.info(()=>"ImmedReconciler: started.");
        val dest  = new TimedGradient(size); 
        var compG:TimedGradient  = new TimedGradient(size); 
        var totalMBReceived:UInt = 0un;
        reducer.initialize(size);
        while (totalMBReceived < maxMB) { 
            compG = fromLearner.get(compG);
            //            reducer.acceptContrib(compG);
            //            reducer.reduceIfPossible(dest, timeStamp);
            val includedMB = dest.loadSize();
            if (includedMB > 0un) { 
                totalMBReceived += includedMB;
                timeStamp++;
                dest.timeStamp = timeStamp;
                learner.acceptGradientFromReconciler(dest);
                dest.setLoadSize(0un);
            }// includeMB>0
        } // while
        logger.info(()=>"Reconciler: Exited main loop, terminating.");
        done.set(true);
    } //reconciler
}
// vim: shiftwidth=4:tabstop=4:expandtab
