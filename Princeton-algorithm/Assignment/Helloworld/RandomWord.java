import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;
public class RandomWord {
    public static void main(String[] args){
        String ret = "";
        double p = 1.0;
        String tem = "";
        while(!StdIn.isEmpty()){
            tem=StdIn.readString();
            if(StdRandom.bernoulli(1/p)){
                ret=tem;
            }
            p++;
        }
        StdOut.println(ret);
    }
}
