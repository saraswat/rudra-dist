package xrudra;

import xrudra.util.Logger;
import xrudra.util.Timer;
import xrudra.util.SwapBuffer;


import x10.util.concurrent.AtomicBoolean;
import x10.util.Team;


public class HardBufferedLearner(maxMB:UInt, noTest:Boolean, 
                                 weightsFile:String) extends Learner {

    public def this(confName:String, noTest:Boolean, 
                    weightsFile:String, mbPerEpoch:UInt, profiling:Boolean, 
                    team:Team, logger:Logger, lt:Int, solverType:String,
                    nLearner:NativeLearner, 
                    maxMB:UInt) {
        super(confName, mbPerEpoch, 0un, profiling, nLearner, team, logger, lt, solverType);
        property(maxMB, noTest, weightsFile);
    }
    val trainTimer = new Timer("Training Time:");
    val weightTimer = new Timer("Weight Update Time:");

    def run(fromLearner:SwapBuffer[TimedGradient], 
            toLearner:SwapBuffer[TimedGradient], done:AtomicBoolean) {
        logger.info(()=>"Learner: started.");
        var compG:TimedGradient = new TimedGradient(size); 
        compG.timeStamp = UInt.MAX_VALUE;
        val testManager = here.id==0? (this as Learner).new TestManager(noTest, solverType) : null;
        if (here.id==0) testManager.initialize();
        epochStartTime= System.nanoTime();
        initWeightsIfNeeded(weightsFile);
        var dest:TimedGradient = new TimedGradient(size); 
        computeGradient(compG);         
        compG=fromLearner.put(compG);
        // first time around, no gradient to receive from network.
        while (!done.get()) {
            computeGradient(compG);         
            compG=fromLearner.put(compG);

            dest=toLearner.get(dest);
            acceptNWGradient(dest);

            if (testManager != null) testManager.touch();
        } // while !done

        if (testManager != null) testManager.finalize();
        logger.info(()=>"Learner: Exited main loop.");
        if (here.id==0) logger.notify(()=> "" + cgTimer);
    } //learner

}
// vim: shiftwidth=4:tabstop=4:expandtab
