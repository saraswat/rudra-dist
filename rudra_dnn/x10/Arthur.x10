// main class orchestrate all the learner agents that learn a model, named after Arthur Watson
import x10.compiler.NonEscaping;
import x10.util.OptionsParser;
import x10.util.Option;
import x10.util.Team;
import x10.util.ArrayList;
import x10.compiler.Uncounted;
public class Arthur {
	public static val PARAM_SERVER_PID:Long = 0l;
	public static val STATS_SERVER_PID:Long = 1l;
	public static val IMG_SERVER_PID:Long = 2l;
	//public static val TEST_SERVER_PID:Long = 0n;
	public static val SERVERNUM:Long = 3l;
	
	public static val HARD_SYNC_LEVEL = 0n;
	
	public static val CHAOTIC_SYNC_LEVEL = 2n;
	public static val SMART_SYNC_LEVEL = 4n;
	
	var learnerPlh:PlaceLocalHandle[LearnerAgentX10];
	val numEpochs:Long;
	val mbPerPlace:Long;
	val mbSize:Long;
	val params:Params;
	val confName:String; // single node configuration file
	var numMBPerEpoch:Long;
	val numLearner:Long;
	val profiling:Boolean;
	val P:Long;
	// grp, team related operations
	var psGroup:SparsePlaceGroup; // parameter server group
	var psTeam: Team;
	
	var laGroup:SparsePlaceGroup; // LA group
	var laTeam: Team;
	
	// added on June 10, 2015, restructre code so that  dont cram everything into LearnerAgent /// 
	var SS: GlobalRef[StatsServer];
	var PS: GlobalRef[ParameterServer];
	// end of restructured code
	public def this(params: Params, snConfigName:String, profiling:Boolean){
	    Console.ERR.println("Arthur constructor.");
		this.P = Place.numPlaces();
		this.numLearner = P- SERVERNUM;
		this.params = params;
		this.numEpochs = params.numEpochs;
		this.mbPerPlace = params.mbPerPlace;
		this.mbSize = params.mbSize;
		this.numMBPerEpoch = params.trainingNum / params.mbSize;
		
		val q = numMBPerEpoch / numLearner;
		val r = numMBPerEpoch % numLearner;
		if(r == 0){
			//this.numMBPerEpoch = tmpMBPerEpoch;
		}else{
			this.numMBPerEpoch = (q+1) * numLearner;
		}
		
		
		this.confName = snConfigName;
		this.profiling = profiling;
		
		/* init ps group        */
		val psgrpplaces:ArrayList[Place] = new ArrayList[Place]();
		for(pl in Place.places()){
			if(isPSGrpID(pl.id)){
				psgrpplaces.add(pl); // sorted
			}
		}
		this.psGroup = new SparsePlaceGroup(psgrpplaces.toRail());
		this.psTeam = new Team(psGroup);
		Console.ERR.println("ps group is created.");
		/* end of init ps group   */
		
		
		/* init la group */
		Console.ERR.println("before creating LA group");
		val laplaces:ArrayList[Place] = new ArrayList[Place]();
		for(pl in Place.places()){
			if(isLearnerAgent(pl.id)){
				laplaces.add(pl); // sorted
			}
		}
		Console.ERR.println("before sparse place group is created");
		this.laGroup = new SparsePlaceGroup(laplaces.toRail());
		Console.ERR.println("before team is created");
		this.laTeam = new Team(laGroup);
		Console.ERR.println("learner agent group is created.");
		/* end of init la group*/
		
		val guy = new Guyasuta(this.confName, this.numLearner);
		val _profiling = this.profiling;
		val _numEpochs = this.numEpochs;
		// step 1 initialize param server 
		this.PS = at(Place(Arthur.PARAM_SERVER_PID)){
			val ps: ParameterServer = new ParameterServer(_profiling, guy, _numEpochs);
			new GlobalRef[ParameterServer](ps)
		};
		Console.ERR.println("PS is created");
		
		
		// step 2 initialize stats server 
		this.SS = at(Place(Arthur.STATS_SERVER_PID)){
			val ss: StatsServer = new StatsServer(_profiling, guy);
			new GlobalRef[StatsServer](ss)
		};
		Console.ERR.println("SS is created");
		
		// step 3 
		val _mbPerPlace = this.mbPerPlace;
		val _numMBPerEpoch = this.numMBPerEpoch;
		this.learnerPlh = PlaceLocalHandle.makeFlat[LearnerAgentX10](psGroup, 
				()=>new LearnerAgentX10(_numEpochs, _mbPerPlace, _numMBPerEpoch, guy, _profiling));
		Console.ERR.println("LAs are created");
	}
	
	/**
	 * @param perf: profiling flag
	 */
	public def init(){
		
		val tmpSS = this.SS;
		val tmpPS = this.PS;
		val tmpLA = this.learnerPlh;
		
		val P = Place.numPlaces();
		val learnerPlh = this.learnerPlh; // quirky stuff about X10	
		val snConfigName = this.confName;
		val numLearner = P- SERVERNUM;
		
		
		at(tmpSS) async{
			tmpSS().init();
		}
		Console.ERR.println("SS is inited async.");
		
		
		// step 2 init PSGrp
		finish {
			at(tmpPS) async{
				tmpPS().init(tmpSS, tmpLA);
			}
			for (place in laGroup) {
				at(place) async {
					tmpLA().init(snConfigName, tmpLA, tmpSS, tmpPS, numLearner);
				}
			}
		}
		Console.ERR.println("PS Grp is inited.");
		
	}
	
	
	public def broadcastWeightsFast():void{
		Console.OUT.println("before broadcast");
		val P = Place.numPlaces();
		val tmpPlh = this.learnerPlh; // quirky stuff about X10
		val tmpPS = this.PS;
		val tmpPsTeam = this.psTeam;
		finish {
			at(PS) async{ // don't forget the async, otherwise no call will be made on LA, thus bcast will hang
				Console.OUT.println("PS before broadcast");
				tmpPS().serializeWeights();
				tmpPsTeam.bcast(tmpPS.home, 
						tmpPS().weights, 0l,
						tmpPS().weights, 0l,
						tmpPS().weights.size);
				Console.OUT.println("PS after broadcast");
			}
			for (p in laGroup) {				
				at(p) async {
					Console.OUT.println("la "+ p.id+ " before broadcast");
					// broadcast weights
					tmpPlh().bcastWeightsTimer.tic();
					tmpPsTeam.bcast(Place(PARAM_SERVER_PID), 
							tmpPlh().weights, 0l,
							tmpPlh().weights, 0l,
							tmpPlh().weights.size);
					Console.OUT.println("la "+ p.id+ " after broadcast");
					tmpPlh().bcastWeightsTimer.toc();
					if(tmpPlh().profiling){
						tmpPlh().bcastWeightsTimer.logRemoteMsg(PARAM_SERVER_PID, 0n, 0n); // not sure if this is the right way to profile bcast in this case. 
					}
					
				}
			}
		}
		Console.OUT.println("after broadcast");
		
	}
	
	@NonEscaping
	public final def isPSGrpID(val id:Long): boolean{
		return isLearnerAgent(id) || isParamServer(id);
	}
	
	
	
	
	
	
	
	
	/**
	 * Synchronous SGD run
	 */
	public def ssgdRun():void{
		val P = Place.numPlaces();
		val learnerPlh = this.learnerPlh; // quirky stuff about X10
		val tmpPS = PS;
		val tmpPsTeam = this.psTeam;
		val multiplier:Float = 1.0f / (P - SERVERNUM ) as float;
		for (epochNum in 1..numEpochs) {
			for (mbNum in 1..mbPerPlace) {
				finish for (p in psGroup) {
					at(p) async {
						val learner = learnerPlh();
						// broadcast weights
						tmpPsTeam.bcast(Place(PARAM_SERVER_PID), 
								learner.weights, 0l,
								learner.weights, 0l,
								learner.weights.size);
						
						if (isLearnerAgent(p.id)) {
						    learner.ensureWeightsLoaded();
							// training process
							learner.selTrainMB(epochNum, mbNum);
							//Console.OUT.println("after MB selection.");
							learner.deSerializeWeights(epochNum, mbNum);
							//Console.OUT.println("after deserialize weights");
							learner.trainOneMB(epochNum, mbNum);
							//Console.OUT.println("after train one MB.");
							val trainErr:Float = learner.trainOneMBErr();
							val delta:Rail[Float] = learner.serUpdates(epochNum, mbNum);
							// talk to P.S. server and update the parameters
							
							
							learner.reportTrainErr(trainErr, epochNum, mbNum);
						}
						
					} // at(p) async 
				} // finish for (p in psGroup)
				
				// apply update at PaTramServer place
				
				at(tmpPS){
					tmpPS().serializeWeights();
				}
			} // for (epochNum in 1..numEpochs)
			at(tmpPS){
				tmpPS().updateLearningRate();
			}
			
		}//for(var i:Int = 0n; i < numEpochs; ++i) ..
	}
	
	
	public def smartRun():void{
		val P = Place.numPlaces();
		val tmpPlh = this.learnerPlh; // quirky stuff about X10
		val tmpPS = PS;
		val numEpochs = this.numEpochs;
		val mbPerPlace = this.mbPerPlace;
		val multiplier:Float = 1.0f / (P - SERVERNUM ) as float;
		this.broadcastWeightsFast();
		// broadcast weights
		val tmpPsTeam = this.psTeam;
		
		finish for (place in Place.places()) {
			if(isLearnerAgent(place.id)) {
				at(place) async {
					//Console.OUT.println("place " + place.id+ " in smart run");
					var finishFlag:boolean = false;
				    for (epochNum in 1..(numEpochs*10)) { // June 22, 2015, for now use imbalance factor of 10
						for (mbNum in 1..mbPerPlace) {
							// training process
							// selTrain data
							// if((epochNum == 1) && (mbNum == 1)){
							// tmpPsTeam.bcast(Place(PARAM_SERVER_PID), 
							// 		tmpPlh().weights, 0l,
							// 		tmpPlh().weights, 0l,
							// 		tmpPlh().weights.size);
							// }
							tmpPlh().selTrainMB(epochNum, mbNum);								
							tmpPlh().trainOneMB(epochNum, mbNum);
							val trainErr = tmpPlh().trainOneMBErr();
							tmpPlh().reportTrainErr(trainErr, epochNum, mbNum);
							
							tmpPlh().pushAndPullWeights(multiplier, epochNum, mbNum);
							if(tmpPlh().checkBillBoard(epochNum, mbNum)){
							 	finishFlag = true; 
							 	break; 
							} 
						}// for my share of minibatch
						// deSerializeWeights
						if(finishFlag){
							break;
						}
						// test one epoch
						//val testErr:Float = learnerPlh().testOneEpoch(epochNum, mbNum);	
					} // for each epoch
				} // at(place) async
			} //if (isLearnerAgent(p))
		} // for each place
		at(tmpPS){
			val uuNum = tmpPS().getUUNum();
			Console.OUT.println("smart uuNum " + uuNum);
		}
	}
	
	
	
	
	public static def isLearnerAgent(val id:Long):Boolean{
		return (id != PARAM_SERVER_PID) && (id != STATS_SERVER_PID) && (id != IMG_SERVER_PID);// && (id != TEST_SERVER_PID);
	}
	
	public static def isParamServer(val id:long):Boolean{
		return (id == PARAM_SERVER_PID);
	}
	public static def main(args:Rail[String]) {
		// Option parser
		val cmdLineParams = new OptionsParser(args,  [
		                                              Option("-h", "help", "Print help messages"),
		                                              Option("-perf", "profiling", "Performance profiling")
		                                              ], 
		                                              [                               
		                                               Option("-f", "config", "Configuration file"),
		                                               Option("-e", "epoch", "Number of epochs"),
		                                               Option("-mb", "minibatch", "Minibatch size"),
		                                               Option("-tn", "trainingnumber", "Training sample number"),
		                                               Option("-sl", "synclevel", "Synchronous SGD level"),
		                                               Option("-dl", "debuglevel", "Debugging level")
		                                               ]);
		val h:Boolean = cmdLineParams("-h", false); // help msg
		val perf:Boolean = cmdLineParams("-perf", false); // performance profiling
		val conf:String = cmdLineParams("-f", "defaults.conf"); // configuration file
		val epoch:Long = cmdLineParams("-e", 50l); // epoch number
		val tn:Long = cmdLineParams("-tn", 60000l); // MNIST dataset training samples number
		val mb:Long = cmdLineParams("-mb", 100n); // minibatch size
		val sl:Int = cmdLineParams("-sl", 0n); // sync level, 
		val dl:Int = cmdLineParams("-dl", 4n); // debugging level
		// Arthur
		val P:Long = Place.numPlaces();
		if (P < 4) {
			Console.ERR.println("./rudra: requires at least 4 processes to run");
			System.setExitCode(1n);
			return;
		}
		Console.ERR.println("before arthur is created");
		val mbPerPlace = (tn / mb / (P - SERVERNUM));
		val params = new Params(epoch, mbPerPlace, mb, tn);
		val art:Arthur = new Arthur(params, conf, perf);
		art.init();
		Console.OUT.println("Arthur is initialized");
		//System.sleep(1000);
		
		if(sl == HARD_SYNC_LEVEL){
			art.ssgdRun();
		}else if(sl == SMART_SYNC_LEVEL){
			art.smartRun();
		}else{
			Console.ERR.println("unrecognized synchronization level " + sl);
		}
	}
}
