import x10.util.concurrent.Lock;
public class ParameterServer {
	val id:Long;
	val guy:Guyasuta;
	val profiling:boolean;
	val applyUpdateTimer:ArthurTimer;
	var SS:GlobalRef[StatsServer];
	var weights:Rail[Float];
    var delta:Rail[Float];
    var deltaGR:GlobalRail[Float];
    val lock:Lock = new Lock();
	////////////////////// PS related defines //////////////////////////
	val PS_UU_NOP_FLAG:Int = 1n;
	val PS_UU_SMB_FLAG:Int = 2n;
	val PS_UU_EPOCH_FLAG:Int = 4n;
	var epochNum:Long = 0;
	var asyncUpdateTimes:Long = 0L; //only param server should  update this number 
	val numEpochs:Long;
	
	////////////////// end of PS related defines ////////////////////////
	
	
	///////////////// remote ptr to LA PLH ///////////////////////////////
	var learnerPlh:PlaceLocalHandle[LearnerAgentX10];
	/////////////// end remote ptr to LA PLH ///////////////////////////
	
	public def this(val profiling:boolean, val guy:Guyasuta, val numEpochs:Long){
		this.id =  x10.xrx.Runtime.hereLong();
		this.profiling = profiling;
		this.guy = guy;
	
		this.numEpochs = numEpochs;
		// create timers
		this.applyUpdateTimer = new ArthurTimer(id, ArthurTimer.EID_APPLY_UPDATES);
		
	}
	
	
	
	public def init(val SS:GlobalRef[StatsServer], val lAPlh: PlaceLocalHandle[LearnerAgentX10]):void{
		this.guy.init();
		NativeUtilsNI.initAsPS();
		this.weights = new Rail[Float](NativeUtilsNI.getNetworkSize());
		this.delta = new Rail[Float](NativeUtilsNI.getNetworkSize());
	this.deltaGR = GlobalRail[Float](this.delta);
		this.SS = SS;
		this.learnerPlh = lAPlh;
	}
	
	
	
	// unified update , added on June 7, 2015
	public def unifiedUpdate(val delta:Rail[Float], val multiplier:Float, pid:Long): void{
		applyUpdateTimer.tic();
		val numUpdates:Long;
		val tmpWeights:Rail[float];
		val uuStatus:Int;
		atomic{
			
			uuStatus = NativeUtilsNI.unifiedUpdate(delta, multiplier);
			if((uuStatus & PS_UU_EPOCH_FLAG) == PS_UU_EPOCH_FLAG){
				tmpWeights = this.weights;
			}else{
				tmpWeights = new Rail[Float](0);
			}
		}
		if((uuStatus & PS_UU_EPOCH_FLAG) == PS_UU_EPOCH_FLAG) {
			this.epochNum++;
			//val learnerPlh = this.learnerPlh;
			Console.OUT.println("[PS unified update] about to issue a test request to SS.");
			Console.OUT.flush();
			val tmpEpochNum = this.epochNum;
			Console.OUT.println("SS home + " + SS.home.id);
			val tmpSS = this.SS;
			at(tmpSS) async {
				Console.OUT.println("[SS] about to do test one epoch");
				Console.OUT.flush();
				tmpSS().testOneEpochSC(tmpEpochNum, -1, tmpWeights);
			}
			if(this.epochNum == this.numEpochs){ // finished everything
				shutdownAllLearners();
			}
		}
		applyUpdateTimer.toc();
		if(this.profiling){
			this.applyUpdateTimer.logLocalMsg(-1l, -1l, pid);
		}
	}
	
	// PS need serializeWeights
	public def serializeWeights():void{
		NativeUtilsNI.serializeWeights(this.weights); // dont need a timer for now, this is at the param side
	}
	
	// added on June 7, 2015, 
	public def shutdownAllLearners():void{
		val tmpPlh = this.learnerPlh;
		val killMsg = new Rail[Int](1);
		killMsg(0) = 1n;
		Console.OUT.println("about to kill everyone.");
		finish for (place in Place.places()) {	
			if(Arthur.isLearnerAgent(place.id)){
				at(place) async {
					Rail.asyncCopy(killMsg, 0, tmpPlh().billBoard, 0,1);
					
				}
			}	
		}
	}
	
	// per epoch
	public def updateLearningRate():void{
		NativeUtilsNI.updateLearningRate(); // probably never needed for learner agent, but only used by 
	}
	
	public def getUUNum():Long{
		return NativeUtilsNI.getUUNum();
	}
}
