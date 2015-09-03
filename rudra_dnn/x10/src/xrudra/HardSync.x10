package xrudra;

import xrudra.util.Logger;
import xrudra.util.Timer;

import x10.util.concurrent.AtomicBoolean;
import x10.util.Team;


public class HardSync(maxMB:UInt) extends Learner {

    public def this(confName:String, mbPerEpoch:UInt, profiling:Boolean, 
                    team:Team, logger:Logger, lt:Int, 
                    nLearner:NativeLearner, 
                    maxMB:UInt) {
        super(confName, mbPerEpoch, 0un, profiling, null, nLearner, team, logger, lt);
        property(maxMB);
    }
    val trainTimer = new Timer("Training Time:");
    val allreduceTimer = new Timer("Allreduce Time:");
    val weightTimer = new Timer("Weight Update Time:");
    def run(reducer:AtLeastRAllReducer) {
        logger.info(()=>"Learner: started.");

        val compG = new TimedGradient(size); 
        compG.timeStamp = UInt.MAX_VALUE;

        val testManager = here.id==0? (this as Learner).new TestManager() : null;
        if (here.id==0) testManager.initialize();
        
        val dest = new TimedGradient(size);
        epochStartTime= System.nanoTime();

        initWeights();
        reducer.initialize(size);

        while (totalMBProcessed < maxMB) {
            computeGradient(compG);         

            allreduceTimer.tic();

            team.allreduce(compG.grad, 0, dest.grad, 0, size, Team.ADD);
            timeStamp++;
            dest.timeStamp=timeStamp;
            compG.setLoadSize(0un);
            allreduceTimer.toc();
            if (here.id==0) 
               logger.notify(()=>"Reconciler: <- Network "  
                             + dest + "(" + allreduceTimer.lastDurationMillis()+" ms)");

            weightTimer.tic();
            acceptNWGradient(dest);
            weightTimer.toc();
            if (testManager != null) testManager.touch();
        } // while !done

        if (testManager != null) testManager.finalize();
        logger.info(()=>"Learner: Exited main loop.");
        if (here.id==0) {
            logger.notify(()=> "" + cgTimer);
            logger.notify(()=> "" + allreduceTimer);
            logger.notify(()=> "" + weightTimer);
        }
    } //learner

}
// vim: shiftwidth=4:tabstop=4:expandtab
