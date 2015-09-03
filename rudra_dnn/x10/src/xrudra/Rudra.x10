package xrudra;

import x10.util.OptionsParser;
import x10.util.Option;
import x10.util.Team;
import x10.util.concurrent.AtomicBoolean;

import xrudra.util.Logger;
import xrudra.util.SwapBuffer;
import xrudra.util.BBuffer;
import xrudra.util.PhasedT;
import xrudra.util.MergingMonitor;
import xrudra.util.Maybe;
/**
 Top-level class for the X10-based deep learner.
 */
public class Rudra(confName:String, profiling:Boolean, spread:UInt, 
                   solverType:String, hardSync:Boolean, desiredR:Int, 
                   nwMode:Int, nwSize:Int, meanFile:String, 
                   ll:Int, lt:Int, lr:Int, lu:Int)  {
    /** Reconciler should drop undelivered gradient when a new network gradient arrives.
     */
    public static val NW_DROP=0n;

    /** Reconciler should accumulate new gradient into undelivered gradient (if any).
     */
    public static val NW_ACCUMULATE=1n;

    /** Reconciler should buffer new gradient. Note: Must have NW_BUFFER < NW_IMMEDIATE < NW_APPLY.
     */
    public static val NW_BUFFER=2n;

    /** Reconciler should apply immediately to the native learner, possibly incurring races.

     */
    public static val NW_IMMEDIATE=3n;

    /** Reconciler should mantain its own native learner and apply new gradient to generate
        new weights.
     */
    public static val NW_APPLY=4n;

    val logger = new Logger(lu);
    val P = Place.numPlaces();
    val group = PlaceGroup.make(P);
    val team = new Team(group);

    public def run():void {
        // global value, can be referenced across places
        val gCount= new GlobalRef[PhasedT[Int]](new PhasedT[Int](0n,-1n)); 
        val atleastR = new AtLeastRAllReducer(desiredR, team, new Logger(lr), gCount);
        val _mmPLH : Maybe[PlaceLocalHandle[MergingMonitor]] = nwMode == NW_APPLY ? 
            new Maybe[PlaceLocalHandle[MergingMonitor]](PlaceLocalHandle.make[MergingMonitor](Place.places(), 
                                                            ()=> new MergingMonitor()))
            : null;

        finish for (p in Place.places()) at(p) async { // this is meant to leak in!!
            val done = new AtomicBoolean(false);
            val nLearner= Learner.makeNativeLearner(confName, solverType, meanFile, false);
            val numEpochs = nLearner.getNumEpochs() as UInt;
            val mbSize = nLearner.getMBSize() as UInt;
            val numTrainingSamples = nLearner.getNumTrainingSamples() as UInt;
            // rounded up to nearest unit
            val mbPerEpoch = ((nLearner.getNumTrainingSamples() + mbSize - 1) / mbSize) as UInt; 
            val maxMB = (numEpochs * mbPerEpoch) as UInt;
            if (here == Place.FIRST_PLACE) {
                logger.emit("Training with "
                    + P + " places over "
                    + numTrainingSamples + " samples, "
                    + numEpochs + " epochs, "
                    + mbPerEpoch + " minibatches per epoch = "
                    + maxMB + " minibatches.");
            }
            val networkSize = nLearner.getNetworkSize();
            val size = networkSize+1;
            if (hardSync) {
                if (here.id==0) logger.info(()=> "Rudra: Starting HardSync");
                new HardSync(confName, mbPerEpoch, profiling, 
                                    team, new Logger(ll), lt, nLearner, 
                                    maxMB).run(atleastR);
                
            } else if (nwMode == NW_APPLY) {
                val mmPLH=_mmPLH();
                val fromLearner = SwapBuffer.make[TimedGradient](true, new TimedGradient(size));
                val nl= Learner.makeNativeLearner(confName, solverType, meanFile, true);
                val localMM = mmPLH();
                val learner=new ApplyLearner(confName, mbPerEpoch, spread, profiling, 
                                             done, localMM,team, new Logger(ll), 
                                             lt, nLearner);
                val ar=new ApplyReconciler(size, maxMB, nl, desiredR, atleastR, 
                                           mmPLH, new Logger(lr));
                async learner.run(fromLearner, ar);
                ar.run(fromLearner, done);              
            } else if (nwMode == NW_IMMEDIATE) {
                val fromLearner = SwapBuffer.make[TimedGradient](true, new TimedGradient(size));
                val learner=new ImmedLearner(confName, mbPerEpoch, spread, profiling, 
                                             done, nLearner, team, new Logger(ll), lt);
                val ir=new ImmedReconciler(size, maxMB, learner, atleastR, new Logger(lr));
                async learner.run(fromLearner);
                ir.run(fromLearner, done);              

            } else {
                throw new Exception("Not implemented yet.");
            }
        }
    }
    
    public static def main(args:Rail[String]) {
        val bootLogger = new Logger(Logger.EMIT);
        bootLogger.emit("Hello, Rudra!");
        // Option parser
        val cmdLineParams = new OptionsParser(args,
            [
                Option("-h", "help", "Print help messages"),
                Option("-hard", "hardsync", "Run in hard sync mode"),
                Option("-perf", "profiling", "Performance profiling")
            ], 
            [                               
                Option("-f", "config", "Configuration file"),
                Option("-a", "allowedSpread", "Allowed spread in a support set (10)"),
                Option("-dl", "debuglevel", "Debugging level (4)"),

                Option("-s", "solver", "Solver (default is SGD)"),
                Option("-r", "atLeastR", "When hardsync is not set, allReduce only when at least R MBs are available (0)"),
                Option("-nwMode", "networkMode", 
                       "Value (DROP=0,ACCUMULATE=1,IMMEDIATE=2,BUFFER=3,APPLY=4) determines reconciler action on arrival of new gradient (4)"),
                Option("-nwSize", "networkBufferSize", "Size of nw buffer, used only with -nwMode 2"),
                Option("-meanFile", "meanFile", "Path to the mean file, used for processing image data, e.g. for imagement"),
                Option("-ll", "logLearner",    "log level (INFO=0,WARN=1,NOTIFY=2,ERROR=3)for Learner (2)"),
                Option("-lt", "logTester",     "log level (INFO=0,WARN=1,NOTIFY=2,ERROR=3)for Tester (2)"),
                Option("-lr", "logReconciler", "log level (INFO=0,WARN=1,NOTIFY=2,ERROR=3)for Reconciler (2)"),
                Option("-lu", "logRudra",      "log level (INFO=0,WARN=1,NOTIFY=2,ERROR=3)for Rudra (2)")
            ]);
        val h:Boolean = cmdLineParams("-h"); // help msg
        if (h) {
            Console.OUT.println(cmdLineParams.usage("Usage:\n"));
            return;
        }
        val perf:Boolean      = cmdLineParams("-perf"); // performance profiling
        val confName:String   = cmdLineParams("-f", "defaults.conf"); // configuration file
        val dl:Int            = cmdLineParams("-dl", 4n); // debugging level TODO: remove
        var spread:UInt       = cmdLineParams("-a", 10un); // allowed spread
        val solverType:String = cmdLineParams("-s", "sgd");// by default the solver type is SGD
        val hardSync:Boolean  = cmdLineParams("-hard");
        var desiredR:Int      = cmdLineParams("-r", 0n);
        var nwMode:Int        = cmdLineParams("-nwMode", NW_APPLY);
        val nwSize:Int        = cmdLineParams("-nwSize", 10n);
        val meanFile:String   = cmdLineParams("-meanFile", null as String);

        val ll:Int = cmdLineParams("-ll", Logger.NOTIFY);
        val lt:Int = cmdLineParams("-lt", Logger.NOTIFY);
        val lr:Int = cmdLineParams("-lr", Logger.NOTIFY);
        val lu:Int = cmdLineParams("-lu", Logger.NOTIFY);


        if (hardSync) {
            if (desiredR > 0)  {
                Console.OUT.println("Both hardsync (-hard) and desiredR (-r)"
                  + " are set. Hardsync takes priority, desiredR set to 0.");
                desiredR=0n;
            }
            if (nwMode >= Rudra.NW_BUFFER)  {
                Console.OUT.println("Hardsync (-hard) is set. "
                + "nwMode is irrelevant, changed to NW_DROP=" + NW_DROP + ".");
                nwMode=NW_DROP;
            }
            if (spread > 1un) {
                Console.OUT.println("Hardsync (-hard) is set."
                                    +" spread is irrelevant, changed to 1");
                spread=1un;
            }
        }

        // echo command line parameters
        bootLogger.emit("rudra" + (perf?" -perf" :"")+ " -f " + confName 
                    + " -a " + spread + " -dl " +dl + " -s " + solverType
                    + (hardSync?" -hard":"") + " -nwMode " + nwMode 
                    + " -nwSize " + nwSize + " -r " + desiredR
                    + " -meanFile " + meanFile
                    + " -ll " + ll + " -lt " + lt + " -lr " + lr + " -lu " + lu);
        val rudra = new Rudra(confName, perf, spread, solverType, hardSync, 
                              desiredR, nwMode, nwSize, meanFile, 
                              ll, lt, lr, lu);
        val startTime = System.currentTimeMillis();
        rudra.run();
        val runTime = (System.currentTimeMillis()-startTime)/1000.0;
        bootLogger.emit("Time=" + runTime + "sec. \n Goodbye!\n");
    }
}
// vim: shiftwidth=4:tabstop=4:expandtab
