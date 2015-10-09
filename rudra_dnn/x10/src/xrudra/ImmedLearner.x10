package xrudra;

import xrudra.util.Logger;
import xrudra.util.Timer;
import xrudra.util.SwapBuffer;

import x10.util.concurrent.AtomicBoolean;
import x10.util.concurrent.Lock;
import x10.util.Team;


public class ImmedLearner(noTest:Boolean) extends Learner {

    public def this(confName:String, noTest:Boolean, mbPerEpoch:UInt, spread:UInt, 
                    profiling:Boolean, 
                    nLearner:NativeLearner, 
                    team:Team, logger:Logger, lt:Int, solverType:String) {
        super(confName, mbPerEpoch, spread, profiling, nLearner, team, logger, lt, solverType);
        property(noTest);
    }

    // method called by reconciler thread.
    val lock = new Lock();
    public def getTotalMBProcessed():UInt {
        try {
            lock.lock();
            return totalMBProcessed;
        } finally {
            lock.unlock();
        }
    }
    def setTimeStamp(ts:UInt):void {
        try {
            lock.lock();
            timeStamp = ts;
        } finally {
            lock.unlock();
        }
    }
    def acceptGradientFromReconciler(g:TimedGradient) {
        val includeMB = g.loadSize();
        try {
            lock.lock();
            totalMBProcessed += includeMB;
            timeStamp = g.timeStamp;
        } finally {
            lock.unlock();
        }
        acceptGradients(g.grad, includeMB);
        logger.info(()=>"Reconciler: delivered network gradient " + g + " to learner.");
    }
    def run(fromLearner:SwapBuffer[TimedGradient], done:AtomicBoolean) {
        logger.info(()=>"Learner: started.");
        var compG:TimedGradient = new TimedGradient(size); 
        compG.timeStamp = UInt.MAX_VALUE;
        val testManager = here.id==0? (this as Learner).new TestManager(noTest, solverType) : null;
        if (testManager != null) testManager.initialize();
        val currentWeight = new TimedWeight(networkSize);
        initWeights();
        while (! done.get()) {
            computeGradient(compG);
            compG=deliverGradient(compG, fromLearner);
            // the reconciler will come in and update weights asynchronously
            if (testManager != null) testManager.touch();
        } // while !done

        if (testManager != null) testManager.finalize();
        logger.info(()=>"Learner: Exited main loop.");
    } //learner

}
// vim: shiftwidth=4:tabstop=4:expandtab

