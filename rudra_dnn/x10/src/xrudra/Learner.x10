package xrudra;

import x10.compiler.NonEscaping;

import xrudra.util.Logger;
import xrudra.util.Timer;
import xrudra.util.SwapBuffer;
import xrudra.util.BBuffer;

import x10.util.concurrent.AtomicBoolean;
import x10.util.Team;
import x10.io.Unserializable;
import x10.compiler.Pinned;

@Pinned public class Learner(confName: String, mbPerEpoch:UInt, spread:UInt, 
                     profiling:boolean, done:AtomicBoolean, 
                     nLearner: NativeLearner, 
                     team:Team, logger:Logger, lt:Int) implements Unserializable {
    
    static public def makeNativeLearner(confName:String, 
                                        solverType:String, 
                                        meanFile:String, isReconciler:Boolean):NativeLearner {
        val nl = new NativeLearner();
        if (meanFile!=null) nl.setMeanFile(meanFile);
        nl.initNativeLand(here.id, confName, Place.numPlaces());
        nl.initAsLA(isReconciler);
        nl.initPSU(solverType);
        return nl; 
    } 
    static def getNetworkSize(nl:NativeLearner):UInt = nl.getNetworkSize() as UInt;

    val startTime = System.nanoTime();
    val id = here.id;
    var totalMBProcessed:UInt = 0un;
    var epoch:UInt = 0un;
    var timeStamp:UInt = 0un;
    val P = Place.numPlaces();
    val networkSize = getNetworkSize(nLearner);
    val size = networkSize+1;
    val cgTimer = new Timer("Compute gradient time:");
    public def getNetworkSize():UInt = getNetworkSize(nLearner);

    public def acceptGradients(delta:Rail[Float], numMB:UInt):void {
        nLearner.acceptGradients(delta,numMB as Long);
    }

    public def serializeWeights(w:Rail[Float]): void{
        nLearner.serializeWeights(w);
    }

    public def deserializeWeights(w:Rail[Float]): void{
        nLearner.deserializeWeights(w);
    }

    public def loadMiniBatch():void {

        nLearner.loadMiniBatch();

    }
    
    public def trainMiniBatch():Float {
        val result = nLearner.trainMiniBatch();
        return result;
    }

    /** If cg already contains gradients, drop them if they are stale.
        Load, or accumulate gradients in cg after training a minibatch.
     */
    public def computeGradient(cg:TimedGradient):void {
            val ts = timeStamp;
            cgTimer.tic();
            // Train!
            loadMiniBatch();             

            val e = trainMiniBatch();
            // Get gradients from native learner, mixing them into old gradients, 
            // if they are not stale
            val stale = (cg.timeStamp+spread < ts);
            if (cg.loadSize() == 0un) {
                cg.timeStamp = timeStamp;
            } else {
                if (stale) {
                    logger.warning(()=>"Learner: dropped old computed gradient " + cg);
                    cg.timeStamp = timeStamp;
                    cg.setLoadSize(0un);
                } else {
                    // Take the min here because compG may have accumulated
                    // gradients from past incarnations.
                    if (ts != cg.timeStamp)
                        logger.info(()=> "Gradient "+cg+" will be mixed with gradient generated at " + ts);
                }
            }
            val cgsz = cg.loadSize();
            assert ((cgsz==0un && cg.timeStamp==ts)||(cgsz>0un && cg.timeStamp+spread>=ts))
                : "Learner: old computed gradient " + cg + " is stale at time " + ts + " and still alive.";
            getGradients(cg.grad);    
            logger.info(()=>"Learner: produced " + cg + " at time=" + ts);
            cgTimer.toc();
            if (here.id==0) 
                logger.notify(()=>"Learner: train error=" + e
                              + " at time=" + ts + "(" + cgTimer.lastDurationMillis()+" ms)");
    }
    public def deliverGradient(cg:TimedGradient, 
                               fromLearner:SwapBuffer[TimedGradient]):TimedGradient {
        // Try to deliver gradients to reconciler.
        val tmp = fromLearner.put(cg);
        val sent = (tmp != cg);
        logger.info(()=>"Learner:->Reconciler " + (sent?"delivered ":"tried to deliver ") + cg);
        if (sent) { // successful delivery! compG now contains junk.
            tmp.setLoadSize(0un);
            tmp.timeStamp = timeStamp;
            return tmp;
        } else {
            if (cg.loadSize()>10un)
                logger.warning(()=>"Learner:*** Reconciler seems unresponsive, unable to deliver " + cg.loadSize() + " times.");
            return cg;
        }
    }
    /**
       Modify the updates rail in place (it may contain garbage), except for
       the last value which tracks the number of mini-batches whose gradients
       have been accumulated in this rail. If it is > EPS, then the native call
       must sum-accumulate the new gradient into the updates rail, and increment
       the last value.
     */
    public def getGradients(updates:Rail[Float]):void {
        if (updates(updates.size-1) > 0.0) {
            nLearner.accumulateGradients(updates);
        } else {
            nLearner.getGradients(updates);
        }
        // increase number of gradients received
        updates(updates.size-1) += 1.0f;
    }

    def acceptNWGradient(g:TimedGradient):void {
        val includeMB = g.loadSize();
        // have received a new incoming gradient from reconciler (guaranteed to have some gradients)
        logger.info(()=>"Learner:<-Reconciler " + g);
        assert includeMB > 0un: "Learner: gradient received from reconciler should not be empty.";
        assert g.timeStamp > timeStamp : "Learner: at " + timeStamp 
            + " received network input at older time " + g.timeStamp;
        timeStamp = g.timeStamp;
        acceptGradients(g.grad, includeMB);
        totalMBProcessed += includeMB;
        logger.info(()=>"Learner: processed network i/p " + g);
    }

    def getTotalMBProcessed():UInt = totalMBProcessed;
    var epochStartTime:Long = 0;
    public class TestManager { // instance created at here.id==0
        val toTester:SwapBuffer[TimedWeight] = SwapBuffer.make[TimedWeight](false, new TimedWeight(networkSize));
        var weights:TimedWeight= new TimedWeight(networkSize);
        var lastTested:UInt=0un;
        def initialize() {
            async new Tester(confName, new Logger(lt)).run(networkSize, toTester);
        }
        def touch():void {
            // At place 0: Test for epoch transition, try to get a Tester to run with these weights
            val thisEpoch = (getTotalMBProcessed()/mbPerEpoch);
            if (here.id == 0 && thisEpoch > epoch) {
                val epochEndTime = System.nanoTime();
                val timeTaken = (epochEndTime-epochStartTime)/(1000.0*1000.0);
                logger.emit(()=>"Learner: Epoch "  + epoch + " took " + timeTaken + " msec");
                epoch = thisEpoch;
                epochStartTime=epochEndTime;
                val w = weights;
                weights.timeStamp = epoch;
                if (toTester.needsData()) {
                    serializeWeights(weights.weight);
                    val result = toTester.put(weights),accepted=result!=weights;
                    if (accepted) lastTested=epoch;
                    logger.info(()=>"Learner: Tester "+(accepted?"accepted " : "did not accept ") + w);
                    weights = result;
                } else {
                    logger.info(()=>"Learner: Tester too busy to accept " + w);
                }
            }
        }
        def finalize() {
            if (lastTested < epoch) { // make sure u test the last weights
                weights.timeStamp=epoch;
                serializeWeights(weights.weight);
                toTester.put(weights);
            }
            toTester.put(TimedWeight.POISON);
        }
    } // TestManager

    def initWeights() {
        // learner 0 broadcast weights, to make sure that we start from the same 
        val ns = networkSize as Long;
        val initW:Rail[Float] = new Rail[Float](ns);
        if(here.id == 0 ){
            serializeWeights(initW); // place zero serialize weights
            team.bcast(Place(0), initW, 0l, new Rail[Float](ns), 0l, ns);
        }else{
            team.bcast(Place(0), initW, 0l, initW, 0l, ns);
            deserializeWeights(initW);
        }
    }
}
// vim: shiftwidth=4:tabstop=4:expandtab
