import edu.princeton.cs.algs4.StdDraw;
import java.util.Comparator;
// import edu.princeton.cs.algs4.StdOut;
// import edu.princeton.cs.algs4.StdRandom;
// import java.util.Arrays;


public class Point implements Comparable<Point> {

    private final int x;
    private final int y;

    public Point(int x, int y){
        this.x = x;
        this.y = y;
    }

    public void draw(){
        StdDraw.point(x,y);
    }

    public void drawTo(Point that){
        StdDraw.line(this.x, this.y, that.x, that.y);
    }

    public String toString(){
        return "(" + x + ", " + y + ")";
    }

    public int compareTo(Point that){
        if (that == null){
            throw new NullPointerException();
        }
        if (this.y == that.y){
            return Integer.compare(this.x, that.x);
        }
        else {
            return Integer.compare(this.y, that.y);
        }
    }

    public double slopeTo(Point that){
        if (that == null){
            throw new NullPointerException();
        }
        if (compareTo(that) == 0) return Double.NEGATIVE_INFINITY;
        if (this.x == that.x)  return Double.POSITIVE_INFINITY;
        if (this.y == that.y) return +0.0;
        return 1.0 * (that.y - this.y) / (that.x - this.x);
    }

    private class Slopeorder implements Comparator<Point>{
        public int compare (Point a, Point b){
            if (a  == null || b == null){
                throw new NullPointerException();
            }
            return Double.compare(slopeTo(a), slopeTo(b));
        }
    }

    public Comparator<Point> slopeOrder() {
        return new Slopeorder();
    }

    // public static void main(String[] args) {
    //     Point a = new Point(1, 1);
    //     Point b = new Point(1, 1);
    //     StdOut.println(a.compareTo(b));
    //     StdOut.println(a.slopeTo(b));
    //     int n = 6;
    //     Point[] p = new Point[n];
    //     for (int i = 0; i< n; i++){
    //         p[i] = new Point(StdRandom.uniformInt(n), StdRandom.uniformInt(n));
    //     }

    //     Arrays.sort(p);
    //     StdOut.println("sort by location");
    //     for(int i = 0; i < n ; i++){
    //         StdOut.print(p[i] + " ");
    //     }
    //     StdOut.println();

    //     Arrays.sort(p, p[0].slopeOrder());
    //     StdOut.println("sort by slope");
    //     for(int i = 0; i < n ; i++){
    //         StdOut.print(p[i] + " ");
    //     }
    //     StdOut.println();
    // }
} 