package xrudra;
public class TimedGradient(size:Long) { // mutated in place, hence fields are vars.
    var timeStamp:UInt=0un;
    var grad:Rail[Float] = new Rail[Float](size);
    def loadSize():UInt=grad(size-1) as UInt;
    def setLoadSize(l:UInt):void{
        grad(size-1)=l as Float;
    }
    def addIn(g:TimedGradient):void {
        assert size==g.size  : "TimedGradients of different sizes?!?!";
        for (i in  0..(size-1)) grad(i) += g.grad(i);
    }
    def calcHash():Float{
        var result:Float=0.0f;
        for (x in grad) result+=x;
        return result/grad.size;
    }
    public def toString():String = "<TG #" + hashCode() + " load="+ calcHash()+",size="+(grad(size-1) as Long)+",time="+timeStamp+">";
}
