package xrudra;

import x10.util.Team;
import x10.util.concurrent.AtomicBoolean;
import xrudra.util.SwapBuffer;
import xrudra.util.Logger;

public class Tester(confName:String, logger:Logger) {
    static val P = Place.numPlaces();

    var weightsPLH:PlaceLocalHandle[Rail[Float]];

    /** Called in a separate async at place 0. Continuously runs (until done), 
        waiting on the toTester SwapBuffer for a timed gradient represented a weight set (with 
        associated time stamp). On receipt, tests against held out data, prints out result
        and goes back to waiting.
        to test against held out data. Does the testing
     */
    public def run(networkSize:Long,toTester:SwapBuffer[TimedWeight]) {
        assert (here.id == 0) : "Tester.run: Can only run this code at place 0";
        assert (toTester != null) : "Tester.run: the toTester swap buffer cannot be null.";

        weightsPLH = PlaceLocalHandle.make[Rail[Float]](Place.places(), ()=> new Rail[Float](networkSize));
        
        var tsWeight:TimedWeight = new TimedWeight(networkSize);
        L: while (true) {
            val tmp = toTester.get(tsWeight);
            if (tmp != tsWeight) {
                tsWeight = tmp;
                if (tsWeight.size == 0 ) break L;
                val startTS = tsWeight.timeStamp, tsw = tsWeight;
                logger.info(()=>"Tester: testing " + tsw); 
                val startTime = System.currentTimeMillis();
                val score = test(tsWeight.weight); // can take a long time
                logger.notify(()=>"Tester:Test score " + score + " at time " 
                              + startTS + "(took " + (System.currentTimeMillis()-startTime) + " ms)");
            }
        } // while done
        logger.info(()=>"Tester: Exited main loop.");
    }

    /* Invoked by a learner that has access to the weights and wishes
       to determine the test error. This is a simple map reduce computation,
       spawn some number of tasks at different places to compute the error
       on their portion of the data, reduce, and return the result.
     */
    public def test(testWeights:Rail[Float]):Float {
        val _nn = new NativeLearner();
        _nn.initNativeLand(here.id, confName, P);

        Rail.copy(testWeights, weightsPLH());
        val root = here;

        val result = finish(Reducible.SumReducer[Float]()) {
            for (p in Place.places()) at(p) async {
                val nn = new NativeLearner();
       
                nn.initNativeLand(here.id, confName, P);
                nn.initNetwork(false);
                nn.initTestSC(p.id, P);
                val weights = weightsPLH();
                Team.WORLD.bcast(root, weights, 0, weights, 0, weights.size);
                var loss:Float=nn.testOneEpochSC(weights);
                offer loss;
            }
        };
       
        return result*1.0f/P;
    }

}
// vim: shiftwidth=4:tabstop=4:expandtab
