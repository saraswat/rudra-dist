import x10.xrx.Runtime;
public class StatsServer {
	val testTimer:ArthurTimer;
	val id:Long;
	var profiling:boolean = false;
	private var loadTestDataFlag:boolean= false;
	val guy:Guyasuta;
	var startTime:Long;
	public def this(val profiling:boolean, val guy:Guyasuta){
	    /* a hack to increase X10_NTHREADS by 4*/
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();
	    Runtime.increaseParallelism();

            /*end of hack */
		this.id =  x10.xrx.Runtime.hereLong();
		this.testTimer = new ArthurTimer(id, ArthurTimer.EID_TEST);
		this.profiling = profiling;
		this.guy = guy;
		this.startTime = System.nanoTime();
		//this.initAsSS();
	}
	
	public def init(): void{
		this.guy.init(); // init the native land
		NativeUtilsNI.initTestSC(); // for now, always assume that we are using sampleclient to load test data
		Console.OUT.println("SS is initialized");
	}
	
	
	private def loadDataAsaWhole():void{
		async{
			NativeUtilsNI.loadTestData();
			atomic{
				loadTestDataFlag = true;
				Console.OUT.println("[SS]: loaded test data.");
			}
		}
	}
	
	private def testOneEpoch(val epochNum:Long, val mbNum:Long, val tmpWeights:Rail[Float]): Float {
		Console.OUT.println("inside testOneEpoch id:" + this.id);
		Console.OUT.flush();
		this.testTimer.tic();
		assert(this.id == Arthur.STATS_SERVER_PID);
		Console.OUT.println("[SS] epochNum:" + epochNum + " about to test");
		// do the actual testing
		when(loadTestDataFlag);
		Console.OUT.println("[DoTest]:"+epochNum+":"+" about to test ");
		val testErr:Float = NativeUtilsNI.testOneEpoch(tmpWeights);
		Console.OUT.println("[DoTest]:"+epochNum+":"+testErr);
		this.testTimer.toc();
		if(this.profiling){
			this.testTimer.logLocalMsg(epochNum, mbNum);
		}
		return testErr;
	}
	
	public def testOneEpochSC(val epochNum:Long, val mbNum:Long, val tmpWeights:Rail[Float]): Float {
		Console.OUT.println("inside testOneEpochSC id:" + this.id);
		Console.OUT.flush();
		this.testTimer.tic();
		assert(this.id == Arthur.STATS_SERVER_PID);
		Console.OUT.println("[SS] epochNum:" + epochNum + " about to test");
		// do the actual testing
	
		Console.OUT.println("[DoTest]:"+epochNum+":"+" about to test ");
		val testErr:Float = NativeUtilsNI.testOneEpochSC(tmpWeights);
		Console.OUT.println("[DoTest]:"+epochNum+":"+testErr);
		this.testTimer.toc();
		if(this.profiling){
			this.testTimer.logLocalMsg(epochNum, mbNum);
		}
		return testErr;
	}
	
	
	public var reportTrainErrTimes:Long = 0;
	public var reportTrainErrRA: float = 0;
	   
	public def _reportTrainErr(val trainErr:Float, pid:Long, e:Long, mbNum:Long){
	    //atomic{
	    /* Console.OUT.println("RTE:"+ pid + ":"+ e + ":" + mbNum + ":" + (System.nanoTime() as double - startTime as double )/1e9 + "SS" ); */
		reportTrainErrTimes++;
		reportTrainErrRA = (reportTrainErrRA * (reportTrainErrTimes-1) + trainErr) / reportTrainErrTimes;
		if(reportTrainErrTimes % 2400 == 0){ // TODO: June 22, 2015
		   Console.OUT.println("Train Err RA:" +  reportTrainErrRA+ "time(s):" + (System.nanoTime() as double - startTime as double )/1e9);
		}
		//}
		/* Console.OUT.println("[TrainErr]"+"ID:"+pid+"\t"+"Epoch:"+e+"\t"+"mbNum:"+mbNum+"\t"+"trainErr:"+trainErr  */
		/* 		+"\t"+ "time(s):"+ (System.nanoTime() as double - startTime as double )/1e9); */
	}
	
	public def _reportTestErr(val testErr:Float, pid:Long, e:Long, mbNum:Long){
		Console.OUT.println("[TestErr]"+"ID:"+pid+"\t"+"Epoch:"+e+"\t"+"mbNum:"+mbNum+"\t"+"testErr:"+testErr 
				+ "\t" + "time(s):"+ (System.nanoTime() as double - startTime as double )/1e9);
	}
	
	
	
	public def hello():void{
		Console.OUT.println("hello from SS");
	}
}
