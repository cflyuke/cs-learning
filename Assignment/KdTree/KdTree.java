import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.Point2D;
import edu.princeton.cs.algs4.RectHV;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

import java.util.ArrayList;


public class KdTree {
    private class Node {
        private Point2D p;
        private Node left, right ;
        private RectHV rect;
        Node (Point2D point, RectHV rect_0) {
            p = point;
            rect = rect_0;
            left = null;
            right = null;
        }
    }
    private Node root;
    private int num ;

    public KdTree() {
        root = null;
        num = 0;
    }

    public boolean isEmpty() {
        return num == 0;
    }

    public int size() {
        return num;
    }

    public void insert(Point2D p) {
        if (p == null) throw new IllegalArgumentException();
        if (contains(p)) return;
        root = put(root , p, true, 0, 0, 1, 1);
        num += 1;
    }

    private Node put(Node x, Point2D p,boolean isXpartition,double xmin,double ymin,double xmax,double ymax) {
        if (x == null) return new Node(p, new RectHV(xmin, ymin, xmax, ymax));
        int cmp = isXpartition ? Double.compare(x.p.x(), p.x()) : Double.compare(x.p.y(), p.y());
        if (cmp < 0) {
            if (isXpartition) xmin = x.p.x() ;
            else ymin = x.p.y();
            x.left = put(x.left, p, !isXpartition, xmin, ymin, xmax, ymax);
        }
        else {
            if (isXpartition) xmax = x.p.x();
            else ymax = x.p.y();
            x.right = put(x.right, p, !isXpartition, xmin, ymin, xmax, ymax);
        }
        return x;
    }

    public boolean contains(Point2D p) {
        if (p == null) throw new IllegalArgumentException();
        return get(root, p, true) != null;
    }

    private Node get(Node x, Point2D p, boolean isXpartition) {
        if (p == null) throw new IllegalArgumentException();
        if (x == null) return null;
        
        if (p.compareTo(x.p) == 0) return x;
        int cmp = isXpartition? Double.compare(x.p.x(), p.x()) : 
        Double.compare(x.p.y(), p.y());
        if (cmp < 0) return get(x.left , p, !isXpartition);
        else return get(x.right, p, !isXpartition);
    }

    public void draw() {
        draw(root, true);
    }

    private void draw(Node x, boolean isXpartition) {
        if (x == null) return ;
        StdDraw.setPenColor(StdDraw.BLACK);
        StdDraw.setPenRadius(0.01);
        StdDraw.point(x.p.x(), x.p.y());
        StdDraw.setPenRadius();
        if (isXpartition) {
            StdDraw.setPenColor(StdDraw.RED);
            StdDraw.line(x.p.x(), x.rect.ymin(), x.p.x(), x.rect.ymax() );
        }
        else {
            StdDraw.setPenColor(StdDraw.BLUE);
            StdDraw.line(x.rect.xmin(), x.p.y(), x.rect.xmax(), x.p.y());
        }
        draw(x.left, !isXpartition);
        draw(x.right, !isXpartition);
    }

    public Iterable<Point2D> range(RectHV rect) {
        if (rect == null) throw new IllegalArgumentException();
        if (isEmpty()) return null;
        ArrayList<Point2D> rangePoint = new ArrayList<>();
        range(root, rect, rangePoint);
        return rangePoint ;
    }
    private void range(Node x, RectHV rect, ArrayList<Point2D> rangePoint) {
         if (rect.contains(x.p)) rangePoint.add(x.p);
         if (x.left != null && rect.intersects(x.left.rect)) range(x.left, rect, rangePoint);
         if (x.right != null && rect.intersects(x.right.rect))  range(x.right, rect, rangePoint);
    }


    public Point2D nearest(Point2D p) {
        if (p == null) throw new IllegalArgumentException();
        if (isEmpty()) return null;
        double mind = Double.POSITIVE_INFINITY;
        NearestPoint ptop = new NearestPoint(root.p, mind);
        nearest(root, p, ptop);
        return ptop.p;
    }
    private class NearestPoint {
        private Point2D p;
        double minD;
        public NearestPoint( Point2D point , double d) {
            p = point;
            minD = d;
        }
    }
    private void nearest(Node x, Point2D p, NearestPoint ptop) {
        double d1 = x.p.distanceSquaredTo(p);
        if (d1 < ptop.minD) {
            ptop.p = x.p;
            ptop.minD = d1;   
        }
        if (x.left == null && x.right != null) {
            if (ptop.minD > x.right.rect.distanceSquaredTo(p))
            nearest(x.right, p, ptop);
        }
        else if (x.left != null && x.right == null) {
            if (ptop.minD > x.left.rect.distanceSquaredTo(p))
            nearest(x.left, p, ptop);
        }
        else if (x.left != null && x.right != null){
            double dleft = x.left.rect.distanceSquaredTo(p);
            double dright = x.right.rect.distanceSquaredTo(p);
            if (Double.compare(dleft, dright) < 0) {
                if (ptop.minD > dleft) nearest(x.left, p, ptop);
                if (ptop.minD > dright) nearest(x.right, p, ptop);
            }
            else {
                if (ptop.minD > dright) nearest(x.right, p, ptop);
                if (ptop.minD > dleft) nearest(x.left, p, ptop);
            }
        }
    }
    public static void main (String[] args) {
        String filename = args[0];
        In in = new In(filename);
        KdTree kdtree = new KdTree();
        while (!in.isEmpty()) {
            double x = in.readDouble();
            double y = in.readDouble();
            Point2D p = new Point2D(x, y);
            kdtree.insert(p);
        }
        kdtree.draw();
        StdDraw.setPenRadius(0.01);
        StdDraw.setPenColor(StdDraw.RED);
        StdDraw.point(0.81, 0.30);
        Point2D nearest = kdtree.nearest(new Point2D(0.81, 0.30));
        StdDraw.point(nearest.x(), nearest.y());
        StdOut.println(nearest);
    }
}