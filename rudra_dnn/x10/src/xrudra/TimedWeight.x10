package xrudra;

/** TimedWeight is the same as TimedGradient except that the payload rail is of networkSize
    rather than networkSize+1. It represents a time stamped set of weights for the NN, together
    with  loadSize that represents the number of MB used to compute this weight.
    @author vj
 */
public class TimedWeight(size:Long) { // mutated in place, hence fields are vars.
    public static val POISON=new TimedWeight(0,0un);
    var timeStamp:UInt=0un;
    var loadSize:UInt=0un;
    var weight:Rail[Float] = new Rail[Float](size);
    def this(size:Long){property(size);}
    def this(size:Long, ls:UInt){property(size); loadSize=ls;}

    def loadSize():UInt=loadSize;
    def setLoadSize(l:UInt):void{
        loadSize=l;
    }
    def calcHash():Float{
        var result:Float=0.0f;
        for (x in weight) result+=x;
        return result/weight.size;
    }
    public def toString():String = "<TW #" + hashCode() + " load="+ calcHash()
                          +",size="+ loadSize + ",time="+timeStamp+">";
}
