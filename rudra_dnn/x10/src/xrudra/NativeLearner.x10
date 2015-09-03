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
    @Native("c++", "new xrudra::NativeLearner()")
    public def this(){}
        
    @Native("c++", "#this->initNativeLand(#id, #confName->c_str(),  #numLearner)")
        public def initNativeLand(id:Long, confName:String, numLearner:Long):void{}

    @Native("c++", "#this->initNetwork(#b)")
        public def initNetwork(b:Boolean):Int{
        return 1n;
    }

    @Native("c++", "#this->initPSU(#mysolverType->c_str())")
    public def initPSU(val mysolverType:String):void {}

    @Native("c++", "#this->setMeanFile(#fn->c_str())")
    public def setMeanFile(fn:String):void {}

    @Native("c++", "#this->getNetworkSize()")
    public def getNetworkSize():Long {
        return 0;
    }
        
    @Native("c++", "#this->initAsLA(#isReconciler)")
        public def initAsLA(isReconciler:Boolean):void{}
        
    @Native("c++", "#this->loadTestData()")
    public def loadTestData():void{}

    @Native("c++", "#this->loadMiniBatch()")
    public def loadMiniBatch():void{}
        
    @Native("c++", "#this->trainMiniBatch()")
    public def trainMiniBatch():float{
            return 0F;
    }

    @Native("c++", "#this->testOneEpoch()")
    public def testOneEpoch():Float{
        return -2.0f;
    }

    @Native("c++", "#this->initTestSC(#placeID, #numLearner)")
    public def initTestSC(placeID:Long, numLearner:Long):void{}

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

    @Native("c++", "rudra::MLPparams::_numEpochs")
    public def getNumEpochs():Long { return -1; }

    @Native("c++", "rudra::MLPparams::_numTrainSamples")
    public def getNumTrainingSamples():Long { return -1; }

    @Native("c++", "rudra::MLPparams::_batchSize")
    public def getMBSize():Long { return -1; }

    // vj TODO: fix    
    @Native("c++", "#this->initTestSC()")
    public def initTestSC():void{}
            
    @Native("c++", "#this->testOneEpoch(#weights->raw)")
    public def testOneEpoch(weights:Rail[Float]):Float{
        return -3.0f;
    }
            
    @Native("c++", "#this->testOneEpochSC(#weights->raw)")
    public def testOneEpochSC(weights:Rail[Float]):Float{
        return -3.0f;
    }
    @Native("c++", "#this->getTestNum()")
    public def getTestNum():long{
	return -1;
    }
}
// vim: shiftwidth=4:tabstop=4:expandtab
