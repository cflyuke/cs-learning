import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;
public class Permutation {
    public static void main(String[] args){
        int n = Integer.parseInt(args[0]);
        if (n == 0 ){
            return;
        }
        RandomizedQueue<String> example = new RandomizedQueue<>();
        int num = 0;
        int k = 0;
        while (!StdIn.isEmpty()){
            String s = StdIn.readString();
            if (num < n){
                example.enqueue(s);
                num ++;
            }
            else {
                if (StdRandom.uniformInt(1, k + 2) <= n){
                    example.dequeue();
                    example.enqueue(s);
                }
            }
            k ++;
        }
        for (String iter : example){
            StdOut.println(iter);
        }
    }
}
