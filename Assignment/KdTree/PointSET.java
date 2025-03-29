import edu.princeton.cs.algs4.RectHV;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.Point2D;
import edu.princeton.cs.algs4.In;

import java.util.ArrayList;
import java.util.TreeSet;

public class PointSET{
    private TreeSet<Point2D> treeSet;

    public PointSET() {
        treeSet = new TreeSet<>();
    }

    public boolean isEmpty() {
        return treeSet.isEmpty();
    }

    public int size () {
        return treeSet.size();
    }

    public void insert (Point2D p) {
        if (p == null) {
            throw new IllegalArgumentException();
        }
        treeSet.add(p);
    }

    public boolean contains(Point2D p) {
        if ( p == null) {
            throw new IllegalArgumentException();
        }
        return treeSet.contains(p);
    }

    public void draw() {
        StdDraw.setPenColor(StdDraw.BLACK);
        for (Point2D p : treeSet) {
            StdDraw.point(p.x(), p.y());
        }
    }

    public Iterable<Point2D> range(RectHV rect) {
        if (rect == null) {
            throw new IllegalArgumentException();
        }
        ArrayList<Point2D> point = new ArrayList<>();
        for (Point2D p: treeSet) {
            if (rect.contains(p)) {
                point.add(p);
            }
        }
        return point;
    }

    public Point2D nearest(Point2D p) {
        if (p == null) {
            throw new IllegalArgumentException();
        }
        double min = Double.POSITIVE_INFINITY;
        Point2D nearestPoint = null;
        for (Point2D temp : treeSet) {
            double distance = p.distanceTo(temp);
            if (distance < min) {
                min = distance;
                nearestPoint = temp;
            }
        }
        return nearestPoint;
    }
    public static void main(String[] args) {
        String filename = args[0];
        In in = new In(filename);
        PointSET brute = new PointSET();
        while(!in.isEmpty()) {
            double x = in.readDouble();
            double y = in.readDouble();
            Point2D p = new Point2D(x, y);
            brute.insert(p);
        }
        brute.draw();
    }
}