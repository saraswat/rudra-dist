package rudra;

import x10.util.Date;
import rudra.util.SwapBuffer;
import rudra.util.Logger;
import rudra.util.Timer;

public class Tester(testerPlace:Place, confName:String, logger:Logger, solverType:String) {
    static val P = Place.numPlaces();
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

    /**
     * Perform inference on the complete test set and return the test error.
     */
    public def test(epoch:Int, testWeights:Rail[Float]):Float {
        val testWeightsGR = new GlobalRail(testWeights);
        val result = at(testerPlace) {
            val weights = new Rail[Float](testWeightsGR.size);
            finish Rail.asyncCopy(testWeightsGR, 0, weights, 0, weights.size);
            val nn = new NativeLearner(here.id);
            nn.initAsTester(here.id, solverType);
            logger.info(()=>"Tester:Starting testing.");
            val res = nn.testOneEpochSC(weights, 1, 0);
            logger.info(()=>"Tester:Starting checkpoint if needed.");
            nn.checkpointIfNeeded(epoch);
            logger.info(()=>"Tester:Checkpoint finished.");
            nn.cleanup();
            return res;
        };
        testWeightsGR.forget();

        return result;
    }

}
// vim: shiftwidth=4:tabstop=4:expandtab
