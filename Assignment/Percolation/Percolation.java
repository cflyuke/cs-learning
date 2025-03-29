import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.WeightedQuickUnionUF;
public class Percolation {
    private int N;
    private int num;
    private boolean [][] grid;
    private WeightedQuickUnionUF tree;
    private WeightedQuickUnionUF Utree;

    //Check boundaries
    private void checkboundary(int row,int col){
        if (row < 1 || row > N || col < 1 || col > N){
            throw new IllegalArgumentException("Row or col argument error!");
        }
        return;
    }

    //Constructor
    public Percolation(int n){
        if (n <= 0){
            throw new IllegalArgumentException("The number is out of range!");
        }
        N = n;
        num = 0;
        grid = new boolean[n][n];
        tree = new WeightedQuickUnionUF(n*n+2);
        Utree = new WeightedQuickUnionUF(n*n+1);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                grid[i][j] = false;
            }
        }
    }

    //Connect the new open site to its neighborhood
    private void treeUpdate(int row,int col){
        checkboundary(row,col);
        if (row == 1){
            tree.union(0,col);
            Utree.union(0,col);
            if (N > 1 && col == 1){
                if (isOpen(1,col+1)){
                    tree.union(1,2);
                    Utree.union(1,2);
                }
            }
            else if ( N > 1 && col == N ){
                if (isOpen(1,col-1)){
                    tree.union(N-1,N);
                    Utree.union(N-1,N);
                }
            }
            else if (N > 1){
                if (isOpen(1,col-1)){
                    tree.union(col-1,col);
                    Utree.union(col-1,col);
                }
                if (isOpen(1,col+1)){
                    tree.union(col,col+1);
                    Utree.union(col,col+1);
                }
            }
            if (N > 1 && isOpen(2,col)){
                tree.union(col,col+N);
                Utree.union(col,col+N);
            }
        }
        if (row == N ){
            tree.union((N-1)*N+col,N*N+1);
            if (N > 1){
                if (col == 1){
                    if (isOpen(N,col+1)){
                        tree.union(N*(N-1)+1,N*(N-1)+2);
                        Utree.union(N*(N-1)+1,N*(N-1)+2);
                    }
                }
                else if (col == N){
                    if (isOpen(N,col-1)){
                        tree.union(N*N-1,N*N);
                        Utree.union(N*N-1,N*N);
                    }
                }
                else {
                    if (isOpen(N,col-1)){
                        tree.union(N*(N-1)+col,N*(N-1)+col-1);
                        Utree.union(N*(N-1)+col,N*(N-1)+col-1);
                    }
                    if (isOpen(N,col+1)){
                        tree.union(N*(N-1)+col,N*(N-1)+col+1);
                        Utree.union(N*(N-1)+col,N*(N-1)+col+1);
                    }
                }
                if (isOpen(N-1,col)){
                    tree.union((N-2)*N+col,(N-1)*N+col);
                    Utree.union((N-2)*N+col,(N-1)*N+col);
                }
            }
        }
        if (row < N && row > 1){
            if (col == 1){
                if (isOpen(row,col+1)){
                    tree.union((row-1)*N+col,(row-1)*N+col+1);
                    Utree.union((row-1)*N+col,(row-1)*N+col+1);
                }
            }
            else if (col == N){
                if (isOpen(row,col-1)){
                    tree.union((row-1)*N+col,(row-1)*N+col-1);
                    Utree.union((row-1)*N+col,(row-1)*N+col-1);
                }
            }
            else {
                if (isOpen(row,col-1)){
                    tree.union((row-1)*N+col,(row-1)*N+col-1);
                    Utree.union((row-1)*N+col,(row-1)*N+col-1);
                }
                if (isOpen(row,col+1)){
                    tree.union((row-1)*N+col,(row-1)*N+col+1);
                    Utree.union((row-1)*N+col,(row-1)*N+col+1);
                }
            }
            if (isOpen(row-1,col)){
                tree.union((row-2)*N+col,(row-1)*N+col);
                Utree.union((row-2)*N+col,(row-1)*N+col);
            }
            if (isOpen(row+1,col)){
                tree.union((row-1)*N+col,row*N+col);
                Utree.union((row-1)*N+col,row*N+col);
            }
        }
    }

    //Open one site
    public void open(int row, int col){
        checkboundary(row, col);
        if (grid[row-1][col-1] == false){
            num++;
            grid[row-1][col-1] = true;
            treeUpdate(row,col);
        }
    }

    //judge the site if open
    public boolean isOpen(int row, int col){
        checkboundary(row,col);
        if ( grid[row-1][col-1] == true ){
            return true;
        }
        else {
            return false;
        }
    }

    //judge the site is open to the top
    public boolean isFull(int row , int col){
        checkboundary(row,col);
        return Utree.find((N*(row-1)+col))==Utree.find(0);
    }

    //calulate the number of open sites
    public int numberOfOpenSites(){
        return num;
    }

    //percolate or not
    public boolean percolates(){
        if (tree.find(0) == tree.find(N*N+1)){
            return true;
        }
        else {
            return false;
        }
    }

    public static void main(String[] args){
        int n = StdIn.readInt();
        Percolation test = new Percolation(n);
        while(!StdIn.isEmpty()){
            int p = StdIn.readInt();
            int q = StdIn.readInt();
            test.open(p,q);
            if (test.percolates()){
                StdOut.println(test.num);
                return;
            }
        } 
    }
}
