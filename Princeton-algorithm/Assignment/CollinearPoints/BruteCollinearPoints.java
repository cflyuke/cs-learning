import java.util.Arrays;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

import java.util.ArrayList;

public class BruteCollinearPoints {
    private ArrayList<LineSegment> segments;

    public BruteCollinearPoints(Point[] points){
        points = validate(points);
        segments = new ArrayList<>() ;
        int len = points.length;
        for (int i = 0; i < len; i++){
            for (int j = i + 1 ;j < len ; j++){
                for (int k = j + 1 ; k < len; k++){
                    for (int s = k + 1; s < len ;s++) {
                        if (Double.compare(points[i].slopeTo(points[j]),points[i].slopeTo(points[k]))==0
                        && Double.compare(points[i].slopeTo(points[k]),points[i].slopeTo(points[s]))==0)
                        {
                            segments.add(new LineSegment(points[i], points[s]));
                        }
                    }
                }
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
        for ( int i = 0; i < len - 1; i++){
            if (copy[i].compareTo(copy[i+1]) == 0) {
                throw new IllegalArgumentException();
            }
        }
        return copy;
    }

    public int numberOfSegments(){
           return segments.size();
    }

    public LineSegment[] segments(){
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
    BruteCollinearPoints collinear = new BruteCollinearPoints(points);
    for (LineSegment segment : collinear.segments()) {
        StdOut.println(segment);
        segment.draw();
    }
    StdDraw.show();
}
}
