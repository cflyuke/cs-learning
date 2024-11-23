
import java.util.ArrayList;

import edu.princeton.cs.algs4.StdOut;
public class Board {
    private int[][] board;
    private int dim;
    private int Ohamming;
    private int Omanhattan;
    private int blankx;
    private int blanky;

    public Board(int[][] tiles) {
        if (tiles == null) {
            throw new IllegalArgumentException();
        }
        dim = tiles.length;
        int ref = 1;
        board = new int[dim][dim];
        for (int i = 0 ; i<dim ; i++ ) {
            for (int j = 0; j< dim ; j++) {
                board[i][j] = tiles[i][j];
                if (board[i][j] == 0 ) {
                    blankx = i;
                    blanky = j;
                }
            }
        }
        for (int i = 0; i < dim ;i++){
            for (int j = 0; j < dim ; j++){
                if (board[i][j] != ref ){
                     if ( i != blankx || j != blanky){
                        Ohamming  ++;
                        Omanhattan = Omanhattan + Math.abs((board[i][j]-1)/dim - i) + Math.abs((board[i][j]-1)%dim - j);
                     }
                }
                ref ++;
            }
        }
        
    }

    public String toString() {
        StringBuilder strBuilder = new StringBuilder();
        strBuilder.append(dim + "\n");
        for (int row = 0; row < dim; row ++){
            for (int col = 0; col < dim ; col ++) {
                strBuilder.append(" "  + board[row][col]);
            }
            strBuilder.append("\n");
        }
        String string = strBuilder.toString();
        return string ;
    }

    public int dimension() {
        return dim;
    }
    public int hamming() {
        return Ohamming;
    }
    public int manhattan() {
        return Omanhattan;
    }
    public boolean isGoal() {
        return Omanhattan == 0;
    }

    public boolean equals(Object y) {
        if (y == this ) return true;
        if (y == null ) return false;
        if (this.getClass() != y.getClass()) return false;
        Board that = (Board)y;
        if (this.dimension() != that.dimension()) return false;
        for (int i = 0 ; i < dim ; i++) {
            for (int j = 0 ;j <dim ;j++) {
                if (this.board[i][j] != that.board[i][j]) {
                    return false ;
                }
            }
        }
        return true;
    }

    public Iterable<Board> neighbors() {
        ArrayList<Board> neighborHood = new ArrayList<> ();
        Board neighbor = change(blankx - 1 , blanky);
        if (neighbor != null)
            neighborHood.add(neighbor);
        neighbor = change(blankx + 1 , blanky);
        if (neighbor != null)
            neighborHood.add(neighbor);
        neighbor = change(blankx , blanky - 1);
        if (neighbor != null)
            neighborHood.add(neighbor);
        neighbor = change(blankx , blanky + 1);
        if (neighbor != null)
            neighborHood.add(neighbor);
        return neighborHood;
    }

    private Board change(int x, int y) {
        if (x < 0 || x > dim - 1 || y < 0 || y > dim -1) {
            return null;
        }
        int t = board[x][y];
        this.board[x][y] = 0;
        this.board[blankx][blanky] = t;
        Board temp = new Board(this.board);
        this.board[x][y] = t;
        this.board[blankx][blanky] = 0;
        return temp;
    }

    public Board twin() {
         if (blanky != 0){
            int t = this.board[0][0];
            this.board[0][0] = this.board[1][0];
            this.board[1][0] = t;
            Board temp = new Board(this.board);
            this.board[1][0] = this.board[0][0];
            this.board[0][0] = t;
            return temp;
         }
         else if (blankx != 0 ){
            int t = this.board[0][0];
            this.board[0][0] = this.board[0][1];
            this.board[0][1] = t;
            Board temp = new Board(this.board);
            this.board[0][1] = this.board[0][0];
            this.board[0][0] = t;
            return temp;
         }
         else {
            int t = this.board[1][0];
            this.board[1][0] = this.board[1][1];
            this.board[1][1] = t;
            Board temp = new Board(this.board);
            this.board[1][1] = this.board[1][0];
            this.board[1][0] = t;
            return temp;
         }
    }
    public static void main(String[] args) {
         int[][] tiles = new int[3][3];
        tiles[0][0] = 1; tiles[0][1] = 2; tiles[0][2] = 3;
        tiles[1][0] = 4; tiles[1][1] = 0; tiles[1][2] = 6;
        tiles[2][0] = 7; tiles[2][1] = 5; tiles[2][2] = 8;
        Board board = new Board(tiles);
        System.out.println(board.toString());
        System.out.println("\n");

        for (Board bd: board.neighbors()) {
            System.out.println(bd);
            System.out.println(bd.manhattan());
        }
        System.out.println(board.isGoal());
        


        int[][] tiles2 = new int[3][3];
        tiles2[0][0] = 1; tiles2[0][1] = 2; tiles2[0][2] = 3;
        tiles2[1][0] = 4; tiles2[1][1] = 5; tiles2[1][2] = 6;
        tiles2[2][0] = 7; tiles2[2][1] = 0; tiles2[2][2] = 8;

        Board board2 = new Board(tiles2);

        StdOut.println(board2.equals(board));
    }

} 