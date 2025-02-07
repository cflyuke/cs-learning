import edu.princeton.cs.algs4.StdStats;
import edu.princeton.cs.algs4.StdRandom;

public class PercolationStats {
    private int N; // the size of the grid
    private int num; //the times of trial
    private double[] result; //record the result of every trial
    
    //constructor
    public PercolationStats(int n, int trials){
        checkReasonable(n,trials);
        N = n;
        num = trials;
        result = new double[trials];
        for (int i =0;i<num;i++){
            result[i] = test(N);
        }
    }

    // return the number of open sites for each trial
    private double test(int n){
        Percolation example = new Percolation(n);
        while (!example.percolates()){
            int row = StdRandom.uniformInt(1,n+1);
            int col = StdRandom.uniformInt(1,n+1);
            if (!example.isOpen(row,col)){
                example.open(row,col);
            }
        }
        double res = (double)example.numberOfOpenSites()/(n*n);
        return res;
    }

    //check input
    private void checkReasonable(int n, int trials){
        if (n <= 0 || trials <= 0){
            throw new IllegalArgumentException("Illegal input !");
        }
        else {
            return ;
        }
    }

    public double mean(){
        return StdStats.mean(result);
    }

    public double stddev(){
        return StdStats.stddev(result);
    }

    public double confidenceLo(){
        return mean() - 1.96*stddev()/Math.sqrt(num);
    }

    public double confidenceHi(){
        return mean() + 1.96*stddev()/Math.sqrt(num);
    }

    public static void main(String[] args){
        Integer n = Integer.valueOf(args[0]);
        Integer trial = Integer.valueOf(args[1]);
        PercolationStats P = new PercolationStats(n, trial);
        System.out.println("mean                    = " + P.mean());
        System.out.println("stddev                = " + P.stddev());
        System.out.println("95% confidence interval = [" + P.confidenceLo() + "," + P.confidenceHi() + "]");
        return ;
    }
}
