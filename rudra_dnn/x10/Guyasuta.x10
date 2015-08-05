// manager of the native land, there is a Guyasuta at every place
public class Guyasuta {
	var id:Long;
	val confName:String;
	val numLearner:Long;
	
	public def this(val confName:String, val numLearner:Long){
		
		this.confName = confName;
		this.numLearner = numLearner;
	
	}
	
	public def init(){
		this.id = x10.xrx.Runtime.hereLong();
		this.initNativeLand();
		this.initNetwork();
		
		
	}
	
	private def initNativeLand(){
		NativeUtilsNI.initNativeLand(this.id, this.confName, numLearner);
	}
	
	private def initNetwork(){
		NativeUtilsNI.initNetwork();
	}
	
	
}