import x10.xrx.Runtime;
public class LearnerAgentX10 {
	
	var numEpochs:Long;
	var myShareOfMiniBatches:Long; // per epoch
	var trainMultiplier:Float; // training multiplier 
	var numMBPerEpoch:Long;
	var weights:Rail[Float];
	var updates:Rail[Float];
	
	var learnerPlh:PlaceLocalHandle[LearnerAgentX10];
	
	val id:Long;
	
	var startTime:Long;
	
	
	val billBoard = GlobalRail[Int](new Rail[Int](1)); // used by P.S. to initiate things are done
	
	// Timers
	var loadDataTimer:ArthurTimer;
	var bcastWeightsTimer:ArthurTimer;
	var deserWeightsTimer:ArthurTimer;
	var selTrainDataTimer:ArthurTimer;
	var pullWeightsTimer:ArthurTimer;
	var trainTimer:ArthurTimer;
	var serUpdatesTimer:ArthurTimer;
	var pushUpdatesTimer:ArthurTimer;
	var reportTrainErrTimer:ArthurTimer;
	var testTimer:ArthurTimer;
	var reportTestErrTimer:ArthurTimer;
	
	var applyUpdateTimer:ArthurTimer; // added on April 6, 2015 
	
	val profiling:boolean;
	
	
	
	var SS:GlobalRef[StatsServer];
	var PS:GlobalRef[ParameterServer];
	var guy:Guyasuta;
	
	
	public def this(val numEpochs:Long, val myShareOfMiniBatches:Long, val numMBPerEpoch:Long, 
		 val guy:Guyasuta, 
			val profiling:boolean){
		this.id = x10.xrx.Runtime.hereLong();
		
		this.numEpochs = numEpochs;
		this.myShareOfMiniBatches = myShareOfMiniBatches;
		this.numMBPerEpoch = numMBPerEpoch;
		
		this.profiling = profiling;
		this.billBoard()(0n) = 0n; // the billboard is started with 0, 1 means should finish
		this.guy = guy;
		this.createTimers();
		
	}
	
	private def createTimers(){
		// init timers
		this.loadDataTimer = new ArthurTimer(id, ArthurTimer.EID_LOAD_DATA);
		this.bcastWeightsTimer = new ArthurTimer(id, ArthurTimer.EID_BCAST_WEIGHTS);
		this.deserWeightsTimer = new ArthurTimer(id, ArthurTimer.EID_DESER_WEIGHTS);
		this.selTrainDataTimer = new ArthurTimer(id, ArthurTimer.EID_SEL_TRAIN_DATA);
		this.pullWeightsTimer = new ArthurTimer(id, ArthurTimer.EID_PULL_WEIGHTS);
		this.trainTimer = new ArthurTimer(id, ArthurTimer.EID_TRAIN);
		this.serUpdatesTimer = new ArthurTimer(id, ArthurTimer.EID_SER_UPDATES);
		this.pushUpdatesTimer = new ArthurTimer(id, ArthurTimer.EID_PUSH_UPDATES);
		this.reportTrainErrTimer = new ArthurTimer(id, ArthurTimer.EID_REPORT_TRAIN_ERR);
		this.testTimer = new ArthurTimer(id, ArthurTimer.EID_TEST);
		this.reportTestErrTimer = new ArthurTimer(id, ArthurTimer.EID_REPORT_TEST_DATA);
		
	}
	
    public def ensureWeightsLoaded() {
	// vj TODO: check that a rail can be sent to C++, and this is the way to send size.
	// may need the underlying raw pointer to be sent
	NativeUtilsNI.ensureLoaded(weights);
    }
	
	/**
	 * Each learner agent initializes and invoke NI calls
	 */
	public def init(configName: String, 
			learnerPlh:PlaceLocalHandle[LearnerAgentX10], 
			val SS:GlobalRef[StatsServer], val PS:GlobalRef[ParameterServer],
			numLearner:Long): void{
				this.learnerPlh = learnerPlh;
				this.SS = SS;
				this.PS = PS;
		this.guy.init();
		Console.OUT.println("" + id + " is being init as learner agent");
		
		
			this.initAsLA();
			// initialize as learner agent
			Console.OUT.println("" + id + " init as learner agent");
			this.weights = new Rail[Float](NativeUtilsNI.getNetworkSize());
			this.updates = new Rail[Float](NativeUtilsNI.getNetworkSize());
		
		startTime = System.nanoTime();
	}
	
	public def reportTrainErr(val trainErr:Float, val epochNum:Long, val mbNum:Long):void{
		val learnerPlh = this.learnerPlh;
		val myPid = here.id;
		this.reportTrainErrTimer.tic();
		//Console.OUT.println("RTE:"+ myPid + ":"+ epochNum + ":" + mbNum + ":" + (System.nanoTime() as double - startTime as double )/1e9 );
		val  tmpSS = this.SS;
		 at(tmpSS) @x10.compiler.Immediate async{
		     tmpSS()._reportTrainErr(trainErr, myPid, epochNum, mbNum);
		 }
		 //Console.OUT.println("RTE:"+ myPid + ":"+ epochNum + ":" + mbNum + ":" + (System.nanoTime() as double - startTime as double )/1e9 );
		 // vj TODO: Determine why this probe is needed.
		 Runtime.probe();
		/* Console.OUT.println("[pid]"+ myPid + " [TrainErr] "+trainErr + "time(s):" + (System.nanoTime() as double - startTime as double )/1e9 ); */
		this.reportTrainErrTimer.toc();
		if(this.profiling){
			this.reportTrainErrTimer.logRemoteMsg(Arthur.STATS_SERVER_PID, epochNum, mbNum);
		}
	}
	
	////////// Auxilary methods ////////////////////////////
	
	/**
	 * Go to param server and grab weights
	 * vj TODO: Where is this being called? Isnt the broadcast stuff being used to get the weights.
	 */
	public def pullWeights(locked:Boolean, val epochNum:Long, val mbNum:Long):void{
		pullWeightsTimer.tic(); // strictly, we need to remove one , b/c broadcast is also using this piece of code for now	
		val learnerPlh = learnerPlh;
		val weightsGr = GlobalRail[Float](this.weights);
		val tmpPS = PS;
		at(PS) {
			
			if(locked) {
				atomic tmpPS().serializeWeights(); // don't forget to serialize weights before sending
			} else {
				tmpPS().serializeWeights(); // don't forget to serialize weights before sending
			}
			finish Rail.asyncCopy(tmpPS().weights, 0, weightsGr, 0, tmpPS().weights.size);
		}
		pullWeightsTimer.toc();
		if(profiling){
			pullWeightsTimer.logRemoteMsg(Arthur.PARAM_SERVER_PID, epochNum, mbNum);
		}
		this.deSerializeWeights(epochNum, mbNum);
	}
	
	//////////////////////////// The below is where NI calls are updated //////////////////////////
	
	public def initAsLA():void{
		loadDataTimer.tic();
		NativeUtilsNI.initAsLA();
		loadDataTimer.toc();
		if(profiling){
			loadDataTimer.logLocalMsg(0n,0n);
		}
	}
	
	public def deSerializeWeights(val epochNum:Long, val mbNum:Long): void{
		this.deserWeightsTimer.tic();
		NativeUtilsNI.deSerializeWeights(this.weights);
		this.deserWeightsTimer.toc();
		if(this.profiling){
			this.deserWeightsTimer.logLocalMsg(epochNum, mbNum);
		}
	}
	
	public def selTrainMB(val epochNum:Long, val mbNum:Long):void{
		this.selTrainDataTimer.tic();
		NativeUtilsNI.selTrainMB();
		this.selTrainDataTimer.toc();
		if(this.profiling){
			this.selTrainDataTimer.logLocalMsg(epochNum, mbNum);
		}
	}
	
	public def trainOneMB(val epochNum:Long, val mbNum:Long):void{
		this.trainTimer.tic();
		NativeUtilsNI.trainOneMB();
		this.trainTimer.toc();
		if(this.profiling){
			this.trainTimer.logLocalMsg(epochNum, mbNum);
		}
		
	}
	
	public def serUpdates(val epochNum:Long, val mbNum:Long):Rail[Float]{
		this.serUpdatesTimer.tic();
		NativeUtilsNI.serUpdates(this.updates);
		this.serUpdatesTimer.toc();
		if(this.profiling){
			this.serUpdatesTimer.logLocalMsg(epochNum, mbNum);
		}
		return this.updates;
	}
	
	public def trainOneMBErr():Float{
		return NativeUtilsNI.trainOneMBErr();
	}
	
	/**
	 * Send weight updates to param server, and copy back updates weights
	 */
	 public def pushAndPullWeights(multiplier:Float, epochNum:Long, mbNum:Long):void {
		pullWeightsTimer.tic();
		val sourceId = this.id;
		val learnerPlh = this.learnerPlh; // don't capture 'this'
		val delta = learnerPlh().serUpdates(epochNum, mbNum);
		
		val weightsGr = GlobalRail[Float](this.weights);
		val tmpPS = this.PS;
		//val deltaGR = at(tmpPS){ tmpPS().deltaGR};
		val deltaGR = GlobalRail[Float](this.updates);
		
		at(tmpPS) {
		    tmpPS().lock.lock();
		    finish Rail.asyncCopy(deltaGR, 0, tmpPS().delta, 0, tmpPS().weights.size);
	 
		    tmpPS().unifiedUpdate(tmpPS().delta, multiplier, sourceId);
			tmpPS().serializeWeights(); // TODO, we can actually do a smart protocol here
			finish Rail.asyncCopy(tmpPS().weights, 0, weightsGr, 0, tmpPS().weights.size);
			tmpPS().lock.unlock();
		}
		pullWeightsTimer.toc();
		if (profiling) {
			pullWeightsTimer.logRemoteMsg(Arthur.PARAM_SERVER_PID, epochNum, mbNum);
		}
		this.deSerializeWeights(epochNum, mbNum);
	}
	
	// used in hardsync
	public def pushUpdates(multiplier:Float, epochNum:Long, mbNum:Long): void{
		pullWeightsTimer.tic();
		val sourceId = this.id;
		val learnerPlh = this.learnerPlh; // don't capture 'this'
		val delta = learnerPlh().serUpdates(epochNum, mbNum);
		val tmpPS = this.PS;
		at(PS) {
			val paramServer = learnerPlh();
			//paramServer.asyncUpdate(delta, multiplier, sourceId);
			tmpPS().unifiedUpdate(delta, multiplier, sourceId);
		}
		pullWeightsTimer.toc();
		if (profiling) {
			pullWeightsTimer.logRemoteMsg(Arthur.PARAM_SERVER_PID, epochNum, mbNum);
		}
		this.deSerializeWeights(epochNum, mbNum);
	}

	public def checkBillBoard(val epochNum:Long, val mbNum:Long):boolean{
	    var result:boolean = false; 
	    testTimer.tic(); // added on June 22, 2015, hack, use testTimer, as its id is 10
	    if(billBoard()(0) == 1n){
		result = true;
	    }
	    testTimer.toc();
	    if (profiling) {
			testTimer.logLocalMsg(epochNum, mbNum);
	    }
	    return result;
	}
	
	
	
	
	
	
}
