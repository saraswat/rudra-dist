package xrudra.util;
import x10.compiler.NonEscaping;

/** A bounded buffer of size N for values of type T. Used to couple 
 * a producer and consumer, where both can progress at different rates
 * but do not wish to get too far ahead of each other.
 * 
 *

 * <p>Implementation Note: uses the (condition, action) model for 
 * application code exposed by xrudra.util.Monitor. 
 * @author vj
 */
public class BBuffer[T](N:Int) {
	protected val data:Rail[T];
	protected var nextVal:Int=0n;
	protected var size:Int=0n;
	protected val monitor = new Monitor();
	protected var name:String;
	protected val logger = new Logger(Logger.WARNING);
	public def this(N:Int){T haszero}{
		this(N,Zero.get[T]());
	}
	public def this(N:Int, t:T) {
		property(N);
		data = new Rail[T](N, t);
	}
	/**
	 * This is the condition for adding an element. May be overridden in subclasses.
	 */
	protected def hasSpace():Boolean = size < N;
	/**
	 * This is the action for adding an element. May be overridden in subclasses.
	 */
	protected def add(t:T):T {
		logger.info("Bbuffer: " + this + " assigning at " + nextVal + " size= " + size);
		var nextSlot:Int = nextVal+size;
		if (nextSlot >= N) nextSlot %=N;
		val result = data(nextSlot);
		data(nextSlot)=t;
		size++;
		logger.info(()=>"BBuffer: " + this + " exiting put. ");
		return result;
	}
	@NonEscaping
	final public def setName(s:String) { name=s; }

	/** Returns the value in the slot into which t is added. This
	    can be recycled.
	 */
	public operator this()=(t:T):T{
		logger.info(()=> t + " ==> " + this);
		return monitor.on[T](()=> hasSpace(),()=>add(t));
	}
	/**
	 * This is the condition for getting an element. May be overridden in subclasses.
	 */
	protected def hasValue():Boolean = size > 0;
	/**
	 * This is the action for getting an element. May be overridden in subclasses.
	 Takes as argument the value t to be placed in the slot whose value is returned.
	 */
	protected def get(t:T):T {
		val result = data(nextVal);
		data(nextVal)=t;
		if (++nextVal >= N) nextVal %= N;
		size--;
		return result;
	}
	public operator this(t:T):T { 
		val result = monitor.on(()=> hasValue(), ()=>get(t));
		logger.info(()=> (this + " ==> " + result));
		return result;
	}

	/** Returns the next value if the buffer is nonempty, else returns null.
	    Does not block, though it does have to acquire the underlying lock, so 
	    does need to wait for the lock to be available. 
	 */
	public def getIfThere(t:T):T
	    = monitor.atomicBlock[T](()=> hasValue()?get(t):t);

	/** Returns true iff the buffer has space. Used by producer to determine
	    if invoking put would block.
	 */
	public def hasSpaceNow():Boolean = monitor.atomicBlock(()=>hasValue());

	protected def awaken() { monitor.awaken(); }
	
	public def toString()="<" +(name == null? typeName() + " #"+hashCode() : name) + " " + size + ">";
}
