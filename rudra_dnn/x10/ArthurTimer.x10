/**
 * timer used to profile x10 code
 */
public class ArthurTimer {
	public  static  val EID_LOAD_DATA:Int =  1n;
	public static  val EID_BCAST_WEIGHTS:Int = 2n;
	public static  val  EID_DESER_WEIGHTS:Int = 3n;
	public static  val EID_SEL_TRAIN_DATA:Int = 4n;
	public static  val EID_PULL_WEIGHTS:Int =5n;
    public static  val EID_TRAIN:Int =6n;
	public static  val	 EID_SER_UPDATES:Int = 7n;
    public static  val EID_PUSH_UPDATES:Int = 8n;
    public static  val EID_REPORT_TRAIN_ERR:Int = 9n;
    public static  val EID_TEST:Int = 10n;
	public static  val EID_REPORT_TEST_DATA:Int = 11n;

	// param server side
    public static  EID_DESER_UPDATES:Int = 12n;
	public static  EID_SUM_UPDATES:Int = 13n;
    public static  EID_APPLY_UPDATES:Int = 14n;
	public static  EID_SER_WEIGHTS:Int = 15n;
	public static  EID_SEND_WEIGHTS:Int = 16n;
	
	public static LOG_SEP:String = ":";
	
	// class memeber fields
	val pid:Long;
	val eid:Int;
	
	var startNanoSec:Long;
	var endNanoSec:Long;
	var delta:double;
	
	public def this(val pid:Long, val eid:Int){
    	this.pid = pid;
    	this.eid = eid;
	}

	public def tic():void{
    	this.startNanoSec = System.nanoTime();
	}

	public def toc():void{
		this.endNanoSec = System.nanoTime();
    	this.delta = getDelta();
	}

	public def getDelta():double{
		return ((endNanoSec - startNanoSec) as double / 1e9 as double) as double;
	}

	public def logLocalMsg(val epochNum:Long, val mbNum:Long) : void{
    	Console.OUT.println("P"+pid+LOG_SEP+"E"+eid+LOG_SEP+delta+LOG_SEP+epochNum+LOG_SEP+mbNum+LOG_SEP);
	}
	
	/**
	 * added on April 6, to log a msg as a local msg using rid. e.g., asyncUpdate timer, even though the activity
	 * is done locally at Param Server, we will log it as if the @param rid has done the activity to get a 
	 * performance breakdown
	 */
	public def logLocalMsg(val epochNum:Long, val mbNum:Long, val rid:Long){
	    Console.OUT.println("P"+rid+LOG_SEP+"E"+eid+LOG_SEP+delta+LOG_SEP+epochNum+LOG_SEP+mbNum+LOG_SEP);
	}

	public def logRemoteMsg(val other:Long, val epochNum:Long, val mbNum:Long) : void{
	    Console.OUT.println("P"+pid+LOG_SEP+"E"+eid+LOG_SEP+delta+LOG_SEP+epochNum+LOG_SEP+mbNum+LOG_SEP);
	}
	
	

}
