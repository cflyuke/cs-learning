import java.util.ArrayList;
import java.util.Arrays;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

public class FastCollinearPoints {
    private ArrayList<LineSegment> segments;

    public FastCollinearPoints(Point[] points) {
            points = validate(points);
            int len = points.length;
            segments = new ArrayList<>();
            if (len < 4) {
                return ;
            }
            Point[] aux ;
            for (Point p : points) {
                aux = Arrays.copyOf(points, len);
                Arrays.sort(aux, p.slopeOrder());
                int j = 1;
                while(j < len){
                    ArrayList<Point> collinear = new ArrayList<>();
                    Double slope = p.slopeTo(aux[j]);
                    int k = j+1;
                    collinear.add(p);
                    collinear.add(aux[j]);
                    while( k < len && Double.compare(slope, p.slopeTo(aux[k]))==0) {
                         collinear.add(aux[k]);
                         k ++ ;
                    }
                    if (collinear.size() >= 4){
                        Point[] endPoints = collinear.toArray(new Point[0]);
                        Arrays.sort(endPoints);
                        if ( p.compareTo(endPoints[0])==0 ){
                            segments.add(new LineSegment(endPoints[0] , endPoints[endPoints.length - 1]));
                        }
                    }
                    j = k;
                }
            }
    }

    private Point[] validate(Point[] points) {
        if (points == null) throw new IllegalArgumentException();
        for (Point p : points) {
             if (p == null) {
                throw new IllegalArgumentException();
             }
        }
        int len = points.length;
        Point[] copy = Arrays.copyOf(points, len);
        Arrays.sort(copy);
        for( int i = 0; i < len - 1; i++){
            if (copy[i].compareTo(copy[i+1]) == 0) {
                throw new IllegalArgumentException();
            }
        }
        return copy;
    }

    public int numberOfSegments() {
        return segments.size();
    }

    public LineSegment[] segments() {
        return segments.toArray(new LineSegment[0]);
    }

    public static void main(String[] args) {

        // read the n points from a file
        In in = new In(args[0]);
        int n = in.readInt();
        Point[] points = new Point[n];
        for (int i = 0; i < n; i++) {
            int x = in.readInt();
            int y = in.readInt();
            points[i] = new Point(x, y);
        }

        // draw the points
        StdDraw.enableDoubleBuffering();
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        for (Point p : points) {
            p.draw();
        }
        StdDraw.show();

        // print and draw the line segments
        FastCollinearPoints collinear = new FastCollinearPoints(points);
        for (LineSegment segment : collinear.segments()) {
            StdOut.println(segment);
            segment.draw();
        }
        StdDraw.show();
    }
}
