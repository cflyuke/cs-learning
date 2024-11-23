import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.MinPQ;
import edu.princeton.cs.algs4.StdOut;

import java.util.ArrayList;
public class Solver {
    private boolean solvable;
    private Node finalNode;

    //将每一个board转换为节点进行存储，主要目的是为了实现comparable接口
    private class Node implements Comparable<Node> {
        private Board nodeBoard;
        private int priority;
        private int move;
        private Node pre;

        public Node(Board initial) {
            this.move = 0;
            this.pre = null;
            this.nodeBoard = initial;
            this.priority = move + this.nodeBoard.manhattan();
        }

        public Node(Board initial, Node preSearchNode) {
            this.nodeBoard = initial;
            this.move = preSearchNode.move + 1;
            this.pre = preSearchNode;
            this.priority = this.move + this.nodeBoard.manhattan();
        }

        public int compareTo(Node that) {
            if (this.priority < that.priority) {
                return -1;
            }
            else if (this.priority > that.priority){
                return 1;
            }
            return 0;
        }
    }



    public Solver(Board initial) {
        if (initial == null) {
            throw new IllegalArgumentException();
        }

        solvable = false;
        MinPQ<Node> boardPQ = new MinPQ<> ();
        MinPQ<Node> boardPQTwin = new MinPQ<> ();

        Node minNode = new Node(initial);
        Node minNodeTwin = new Node(initial.twin());
        if (minNode.nodeBoard.isGoal()) {
            solvable = true;
            finalNode = minNode;
            return;
        }
        if (minNodeTwin.nodeBoard.isGoal()) {
            solvable = false;
            return;
        }

        while(true) {

            for (Board neighbor : minNode.nodeBoard.neighbors()) {
                if (minNode.pre == null || !neighbor.equals(minNode.pre.nodeBoard) ) {
                    boardPQ.insert(new Node(neighbor, minNode));
                }
            }
            for (Board neighbor : minNodeTwin.nodeBoard.neighbors()){
                if (minNodeTwin.pre == null || !neighbor.equals(minNodeTwin.pre.nodeBoard)) {
                    boardPQTwin.insert(new Node(neighbor, minNodeTwin));
                }
            }

            minNode = boardPQ.delMin();
            minNodeTwin = boardPQTwin.delMin();

            if (minNode.nodeBoard.isGoal()) {
                solvable = true;
                break;
            }

            if (minNodeTwin.nodeBoard.isGoal()) {
                solvable = false;
                break;
            }
        }
        finalNode = minNode;
    }

    public boolean isSolvable() {
        return solvable;
    }

    public int moves() {
        if (solvable)
            return finalNode.move;
        else 
            return -1; 
    }

    public Iterable<Board> solution() {
        if (!solvable) {
            return null;
        }
        ArrayList<Board> solve = new ArrayList<>();
        Node point = finalNode;
        while( point != null) {
            solve.add(point.nodeBoard);
            point = point.pre;
        }
        for (int i = 0, j = solve.size() - 1; i < j; i++, j--) {
            Board temp = solve.get(i);
            solve.set(i, solve.get(j));
            solve.set(j, temp);
        }
        return solve;
    }

    public static void main(String[] args) {
        // create initial board from file
    In in = new In(args[0]);
    int n = in.readInt();
    int[][] tiles = new int[n][n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            tiles[i][j] = in.readInt();
    Board initial = new Board(tiles);

    // solve the puzzle
    Solver solver = new Solver(initial);

    // print solution to standard output
    if (!solver.isSolvable())
        StdOut.println("No solution possible");
    else {
        StdOut.println("Minimum number of moves = " + solver.moves());
        for (Board board : solver.solution())
            StdOut.println(board);
    }
    }
}
