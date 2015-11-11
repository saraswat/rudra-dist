package rudra;

import x10.util.Team;
import x10.util.Date;
import rudra.util.SwapBuffer;
import rudra.util.Logger;
import rudra.util.Timer;

public class Tester(confName:String, logger:Logger, solverType:String) {
    static val P = Place.numPlaces();
    val testerTeam = new Team(PlaceGroup.make(P));
    var weightsPLH:PlaceLocalHandle[Rail[Float]];

    /** Called in a separate async at place 0. Continuously runs (until done), 
        waiting on the toTester SwapBuffer for a timed gradient represented a weight 
        set (with  
        associated time stamp). On receipt, tests against held out data, prints out result
        and goes back to waiting.
        to test against held out data. Does the testing
     */
    public def run(networkSize:Long,toTester:SwapBuffer[TimedWeightWRuntime]) {
        assert (here.id == 0) : "Tester.run: Can only run this code at place 0";
        assert (toTester != null) : "Tester.run: the toTester swap buffer cannot be null.";

        weightsPLH = PlaceLocalHandle.make[Rail[Float]](Place.places(), ()=> new Rail[Float](networkSize));
        
        var tsWeight:TimedWeightWRuntime = new TimedWeightWRuntime(networkSize);
        L: while (true) {
            logger.info(()=>"Tester: Waiting for input");
            val tmp = toTester.get(tsWeight);
            if (tmp != tsWeight) {
                logger.info(()=>"Tester: Received " + tmp);
                tsWeight = tmp;
                if (tsWeight.size == 0 ) break L;
                val epoch = tsWeight.timeStamp, tsw = tsWeight;
                logger.info(()=>"Tester: testing " + tsw); 
                val startTime = System.currentTimeMillis();
                val epochTime = tsWeight.runtime;
                val score = test(epoch as Int, tsWeight.weight); // can take a long time
                logger.notify(()=>new Date() + " Tester:Epoch " + epoch + " took " + Timer.time(epochTime) 
                              + " scored " + score 
                              + " (testing took " + Timer.time(System.currentTimeMillis()-startTime) + ")");
            }
        } // while done
        logger.info(()=>"Tester: Exited main loop.");
    }

    /* Invoked by a learner that has access to the weights and wishes
       to determine the test error. This is a simple map reduce computation,
       spawn some number of tasks at different places to compute the error
       on their portion of the data, reduce, and return the result.
     */
    public def test(epoch:Int, testWeights:Rail[Float]):Float {
        Rail.copy(testWeights, weightsPLH());
        val root = here;
        val team = testerTeam;
        val result = finish(Reducible.SumReducer[Float]()) {
            //            logger.info(()=>"Tester.test: In collecting finish for epoch " + epoch);
            for (p in Place.places()) at(p) async {
                    //                    logger.info(()=>"Tester.test: entered collecting finish async for " + epoch);       
                val weights = weightsPLH();
                //                logger.info(()=>"Tester.test: receiving weights for " + epoch);       
                team.bcast(root, weights, 0, weights, 0, weights.size);
                //                logger.info(()=>"Tester.test: ...received. Creating learner..  " + epoch);       
                val nn = new NativeLearner(here.id);
                nn.initAsTester(here.id, solverType);
                //                logger.info(()=>"Tester.test: ...created. Testing...  " + epoch);       
                offer nn.testOneEpochSC(weights, P);
                //                logger.info(()=>"Tester.test: ...tested and offered for " + epoch);       
                if (p == root) {
                      logger.info(()=>"Tester:Starting checkpoint if needed.");
                    if (true) nn.checkpointIfNeeded(epoch);
                    logger.info(()=>"Tester:Checkpoint finished.");
                }
                nn.cleanup();
            }
        };
        logger.info(()=>"Tester.test: have result for " + epoch);       
        return result*1.0f/P;
    }

}
// vim: shiftwidth=4:tabstop=4:expandtab
