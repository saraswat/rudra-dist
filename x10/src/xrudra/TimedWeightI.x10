package xrudra;

public interface TimedWeightI {

    def loadSize():UInt;
    def setLoadSize(u:UInt):void;
    def timeStamp():UInt;
    def setTimeStamp(u:UInt):void;

    def weightRail():Rail[Float];
}