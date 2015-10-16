package rudra.util;

public class Maybe[T] {
    val t:T;
    public def this(t:T) { this.t=t;}

    public operator this():T=t;
}  
