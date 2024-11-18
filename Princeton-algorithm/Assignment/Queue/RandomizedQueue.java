import java.util.Iterator;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdOut;

public class RandomizedQueue<Item> implements Iterable<Item> {
    private static final int INIT_CAPACITY = 8;
    private Item[] v;
    private int n;
    private int first;
    private int last;

    public RandomizedQueue(){
        v = (Item[]) new Object[INIT_CAPACITY];
        n = 0;
        first = 0;
        last = 0;
    }

    public boolean isEmpty(){
        return n == 0;
    }

    public int size(){
        return n;
    }

    private void resize(int capacity){
        Item[] copy = (Item[]) new Object[capacity];
        for (int i = 0;i<n;i++){
            copy[i] = v[(first+i)%v.length];
        }
        v = copy;
        first = 0;
        last = n ;
    }

    public void enqueue(Item item){
        if (item == null ){
            throw new IllegalArgumentException("Illegal Argument !");
        }
        if (n == v.length){
            resize(2*v.length);
        }
        v[last++] = item;
        n ++;
        if (last == v.length){
            last = 0;
        }
    }

    public Item dequeue(){
        if (isEmpty()){
            throw new java.util.NoSuchElementException("Empty array");
        }
        int id = (first + StdRandom.uniformInt(n)) % v.length;
        Item ret = v[id];
        v[id] = v[first];
        v[first++] = null;
        n --;
        if (first == v.length){
            first = 0;
        }
        if (n > 0 && n == v.length/4){
            resize(v.length/2);
        }
        return ret;
    }
    
    public Item sample(){
        if (isEmpty()){
            throw new java.util.NoSuchElementException("Empty array");
        }
        int id = (first + StdRandom.uniformInt(n)) % v.length;
        return v[id];
    } 
    
    public Iterator<Item> iterator(){
        return new RQueue();
    }
    
    private class RQueue implements Iterator<Item>{
        private int pos;
        private int[] index;

        public RQueue(){
            pos = 0;
            index = new int[n];
            for (int i = 0; i<n; i++){
                index[i] = (i + first) % v.length;
            }
            StdRandom.shuffle(index);
        }

        public boolean hasNext(){
            return pos < n;
        }

        public Item next(){
            if (!hasNext()){
                throw new java.util.NoSuchElementException("No Element !");
            }
            return v[index[pos ++]];
        }

        public void remove(){
            throw new UnsupportedOperationException("You con't remove !");
        }
    }

    public static void main(String[] args){
        RandomizedQueue<Integer> queue = new RandomizedQueue<Integer>();
        for (int i = 0; i < 5; i++){
            queue.enqueue(i);
        }
        StdOut.println(queue.size());
        for (int pos : queue){
            for (int ptr : queue){
                StdOut.print( pos + " and " + ptr + " ");
            }
            StdOut.print("\n");
        }
        for (int i = 0; i < 5 ; i ++){
            queue.dequeue();
        }
        if (queue.isEmpty()){
            StdOut.println("Good job!");
        }
    }
}
