package xrudra;

import x10.compiler.Native;
import x10.compiler.NativeCPPInclude;
import x10.compiler.NativeCPPCompilationUnit;

import xrudra.util.Logger;

@NativeCPPInclude("NativeUtilsImpl.h")
@NativeCPPCompilationUnit("NativeUtilsImpl.cpp")
public class NativeUtilsNI {
	public static def rail2Ptr(r:Rail[Float], name:String){
		
	}
	
	@Native("c++", "::ptr2Rail()")
	public static def ptr2Rail():Rail[Float]{
		return null;
	}
	
	@Native("c++", "::ensureLoaded((#weights)->raw)")
	public static def ensureLoaded(weights:Rail[Float]):void { }

	@Native("c++", "::initNativeLand(#id, #confName->c_str(), #numLearner)")
	    public static def initNativeLand(val id:Long, val confName:String, val numLearner:Long):void{}


	@Native("c++", "::initNetwork()")
	public static def initNetwork():Int{
	    return 1n;
    }

	@Native("c++", "::getNetworkSize()")
	public static def getNetworkSize():Long {
	    return 0;
    }
	
	@Native("c++", "::initAsLA()")
	public static def initAsLA():void{
		
	}
	
	@Native("c++", "::initAsSS()")
	public static def initAsSS():void{
		
	}

	@Native("c++", "::initAsPS()")
	public static def initAsPS():void{
		
	}
	
	@Native("c++", "::loadTrainData()")
	public static def loadTrainData():void{

	}

	@Native("c++", "::loadTestData()")
	public static def loadTestData():void{

	}

	@Native("c++", "::loadMiniBatch()")
	public static def loadMiniBatch():void{
		
	}
	
	@Native("c++", "::trainMiniBatch()")
	public static def trainMiniBatch():float{
	    return 0F;
	}
	
    //    @Native("c++", "::seralizeUpdates(#updates->raw)")
    //	public static def serializeUpdates(updates:Rail[Float]):void { }

	
    /*	@Native("c++", "::testOneEpoch()")
	public static def testOneEpoch():Float{
		return -2.0f;
	}

    
	@Native("c++", "::syncSum(#delta->raw, #multiplier)")
	public static def syncSum(val delta:Rail[Float], val multiplier:Float):void{

	}

	@Native("c++", "::syncUpdate()")
	public static def syncUpdate():void{}

	@Native("c++", "::asyncUpdate(#delta->raw, #multiplier)")
	    public static def asyncUpdate(delta:Rail[Float], multiplier:Float):void{}
    */
	@Native("c++", "::getGradients(#weights->raw)")
	public static def getGradients(weights:Rail[Float]):void {}

	@Native("c++", "::deserializeWeights(#weights->raw)")
	public static def deserializeWeights(weights:Rail[Float]):void {}

	@Native("c++", "::updateLearningRate()")
	public static def updateLearningRate():void{}

        @Native("c++", "::unifiedUpdate(#delta->raw, #multiplier)")
	    public static def unifiedUpdate(val delta:Rail[Float], val multiplier:Float):Int{
	    return -1n;
	}
    /*
	@Native("c++", "::getUUNum()")
	    public static def getUUNum():Long{
	    return -1;
	}
    */
    /*
    // vj TODO: fix    
	    @Native("c++", "::initTestSC()")
	    public static def initTestSC():void{
	    	
	    }
	    
	    @Native("c++", "::testOneEpoch(#weights->raw)")
	    public static def testOneEpoch(val weights:Rail[Float]):Float{
	    	return -3.0f;
	    }
	    
	    @Native("c++", "::testOneEpochSC(#weights->raw)")
	    public static def testOneEpochSC(val weights:Rail[Float]):Float{
	    	return -3.0f;
	    }
    */
	/////////// end of SS section ////////////////////////////////
}
