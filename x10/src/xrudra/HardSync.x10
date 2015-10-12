package xrudra;

import xrudra.util.Logger;
import xrudra.util.Timer;

import x10.util.concurrent.AtomicBoolean;
import x10.util.Team;
/** HardSync implements SGD in parallel by dividing the mini batch evenly across
    all learners, and allreducing the gradients. All learners see the same
    sequence of weights, and weights(t+1) is built from gradients computed
    from weights(t). 
    @author vj
 */
public class HardSync(maxMB:UInt, noTest:Boolean, weightsFile:String) extends Learner {
    public def this(confName:String, noTest:Boolean, 
                    weightsFile: String, mbPerEpoch:UInt, profiling:Boolean, 
                    team:Team, logger:Logger, lt:Int, solverType:String,
                    nLearner:NativeLearner, 
                    maxMB:UInt) {
        super(confName, mbPerEpoch, 0un, profiling, nLearner, team, logger, lt, solverType);
        property(maxMB, noTest, weightsFile);
    }
    val trainTimer     = new Timer("Training Time:");
    val reduceTimer = new Timer("Reduce Time:");
    val bcastTimer = new Timer("BCastTime:");
    val weightTimer    = new Timer("Weight Update Time:");
    def run() {
        logger.info(()=>"Learner: started.");
        val compG = new TimedGradient(size); 
        compG.timeStamp = UInt.MAX_VALUE;
        val testManager = here.id==0? (this as Learner).new TestManager(noTest, solverType) : null;
        if (here.id==0) testManager.initialize();
        val dest = new TimedGradient(size);
        epochStartTime= System.nanoTime();
        initWeightsIfNeeded(weightsFile); 
        var currentEpoch:UInt = 0un;
        while (totalMBProcessed < maxMB) {
            computeGradient(compG);         
            //            team.allreduce(compG.grad, 0, dest.grad, 0, size, Team.ADD);
                reduceTimer.tic();
            team.reduce(Place(0), compG.grad, 0, dest.grad, 0, size, Team.ADD);
                reduceTimer.toc();
                bcastTimer.tic();
            team.bcast(Place(0), dest.grad, 0, dest.grad, 0, dest.grad.size);
                bcastTimer.toc();
            compG.setLoadSize(0un);
            timeStamp++;
            dest.timeStamp=timeStamp;
            if (here.id==0) 
               logger.notify(()=>"Reconciler: <- Network "  
                             + dest + "(" + reduceTimer.lastDurationMillis()+" ms + " 
                             + bcastTimer.lastDurationMillis() + " ms)");
            weightTimer.tic();
            acceptNWGradient(dest);
            weightTimer.toc();
            // follow learning rate schedule given in config file
            if ((totalMBProcessed / mbPerEpoch) > currentEpoch) {
                nLearner.updateLearningRate(++currentEpoch);
            }
            if (testManager != null) testManager.touch();
        } // while !done
        if (testManager != null) testManager.finalize();
        logger.info(()=>"Learner: Exited main loop.");
        if (here.id==0) {
            logger.notify(()=> "" + cgTimer);
            logger.notify(()=> "" + reduceTimer);
            logger.notify(()=> "" + bcastTimer);
            logger.notify(()=> "" + weightTimer);
        }
    } //run
}
// vim: shiftwidth=4:tabstop=4:expandtab
