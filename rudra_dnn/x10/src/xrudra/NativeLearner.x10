package xrudra;

import x10.compiler.Native;
import x10.compiler.NativeCPPInclude;
import x10.compiler.NativeCPPCompilationUnit;
import x10.compiler.NativeRep;

/**
 * Container class for a native learner.
 */
@NativeCPPInclude("NativeLearner.h")
@NativeCPPCompilationUnit("NativeLearner.cpp")
@NativeRep("c++", "xrudra::NativeLearner*", "xrudra::NativeLearner", null)
public class NativeLearner {

    @Native("c++", "NativeLearner::setLoggingLevel(#level)")
        public static def setLoggingLevel(level:Int):void {}

    @Native("c++", "NativeLearner::setJobDir(#jobdir->c_str())")
    public static def setJobDir(jobdir:String):void {}

    @Native("c++", "NativeLearner::setAdaDeltaParams(#rho, #epsilon, #defaultRho, #defaultEpsilon)")
        public static def setAdaDeltaParams(rho:Float, epsilon:Float, 
                                            defaultRho:Float, defaultEpsilon:Float):void {}

    @Native("c++", "NativeLearner::setMeanFile(#fn->c_str())")
    public static def setMeanFile(fn:String):void {}

    @Native("c++", "NativeLearner::setSeed(#id, #seed, #defaultSeed)")
        public static def setSeed(id:Long, seed:Int, defaultSeed:Int):void {}

    @Native("c++", "NativeLearner::setMoM(#mom)")
    public static def setMoM(mom:Float):void {}

    @Native("c++", "NativeLearner::setLRMult(#mult)")
        public static def setLRMult(mult:Float):void {}

    @Native("c++", "NativeLearner::setWD()")
    public static def setWD():void {}

    @Native("c++", "NativeLearner::initFromCFGFile(#cfgName->c_str())")
    public static def initFromCFGFile(cfgName:String):void {}

    @Native("c++", "new xrudra::NativeLearner(#id)")
        public def this(id:Long){}
        
    @Native("c++", "#this->initNativeLand(#id, #confName->c_str(), #seed, #defaultSeed, #numLearner)")
        public def initNativeLand(id:Long, confName:String, numLearner:Long):void{}

    @Native("c++", "#this->checkpointIfNeeded(#epoch)")
    public def checkpointIfNeeded(epoch:Int):void {}

    @Native("c++", "#this->getNetworkSize()")
    public def getNetworkSize():Long {
        return 0;
    }
        

    @Native("c++", "#this->initAsLearner(#weightsFile->c_str(), #solverType->c_str())")
    public def initAsLearner(weightsFile:String, solverType:String):void { }

    @Native("c++", "#this->initAsTester(#placeId, #solverType->c_str())")
    public def initAsTester(placeId:Long, solverType:String):void { }
        
    @Native("c++", "#this->trainMiniBatch()")
    public def trainMiniBatch():float{
            return 0F;
    }

    @Native("c++", "#this->getGradients(#gradients->raw)")
    public def getGradients(gradients:Rail[Float]):void {}

    @Native("c++", "#this->accumulateGradients(#gradients->raw)")
    public def accumulateGradients(gradients:Rail[Float]):void {}

    @Native("c++", "#this->serializeWeights(#weights->raw)")
    public def serializeWeights(weights:Rail[Float]):void {}

    @Native("c++", "#this->deserializeWeights(#weights->raw)")
    public def deserializeWeights(weights:Rail[Float]):void {}

    @Native("c++", "#this->updateLearningRate(#curEpochNum)")
    public def updateLearningRate(val curEpochNum:Long):void{}

    @Native("c++", "#this->acceptGradients(#delta->raw, #numMB)")
    public def acceptGradients(val delta:Rail[Float], val numMB:Long):void{}

    @Native("c++", "#this->getNumEpochs()")
    public def getNumEpochs():Long { return -1; }

    @Native("c++", "#this->getNumTrainingSamples()")
    public def getNumTrainingSamples():Long { return -1; }

    @Native("c++", "#this->getMBSize()")
    public def getMBSize():Long { return -1; }
            
    @Native("c++", "#this->testOneEpochSC(#weights->raw, #numTesters)")
        public def testOneEpochSC(weights:Rail[Float], numTesters:Long):Float {
        return -3.0f;
    }

    /**
     * Free all native-allocated memory.  Afterwards, this object is no
     * longer valid and no further method invocations should be made.
     */
    @Native("c++", "#this->cleanup()")
    public def cleanup():void { }
}
// vim: shiftwidth=4:tabstop=4:expandtab
