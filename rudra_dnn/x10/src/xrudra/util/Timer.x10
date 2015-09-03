package xrudra.util;

/**
 * A simple timer that represents a sequence of disjoint intervals by
 * their number and the sum of their durations.

 * TODO: Maintain a timed sequence, and generate statistics.

 * @author vj
 */

public class Timer(name:String) {

    var count:UInt;
    var duration:Long;
    var lastStart:Long=0;
    var lastEnd:Long=0;

    /** Call tic() to start an interval, and toc() to finish an interval.
     */
    public def tic():void{
        lastStart = System.nanoTime();
    }

    public def toc():void{
        lastEnd = System.nanoTime();
        addDuration(lastDuration());
    }

    public def lastDuration():Long= lastEnd-lastStart;
    public def lastDurationMillis():Long= lastDuration()/(1000*1000);
    /** Add an externally taken measurement to this timer.
     */
    public def addDuration(d:Long):void {
        count++;
        duration +=d;
    }
    public def durationMillis():Long = duration / (1000*1000);
    public def toString():String = "<" + name + " " + count 
                          + " " + durationMillis() + " ms" 
                          + " avg=" + (durationMillis()/(count as Long)) + ">";

}
