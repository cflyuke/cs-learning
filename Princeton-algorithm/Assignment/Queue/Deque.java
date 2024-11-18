import  java.util.Iterator;
import  edu.princeton.cs.algs4.StdOut;
public class Deque<Item> implements Iterable<Item> {
    private class Node{
        Item item;
        Node next;
        Node prev;
    }
    private Node first ;
    private Node last ;
    private int length ;

    public Deque(){
        first = null;
        last = null;
        length = 0;
    }

    public boolean isEmpty(){
        return length == 0;
    }

    public int size(){
        return length;
    }

    private void check_item(Item item){
        if (item == null){
            throw new IllegalArgumentException("invalid item !");
        } 
    }

    public void addFirst(Item item){
        check_item(item);
        Node oldFirst = first;
        first = new Node();
        first.item = item;
        first.next = oldFirst;
        first.prev = null;
        if (isEmpty()){
            last = first;
        }
        else {
            oldFirst.prev = first;
        }
        length ++;
    }

    public void addLast(Item item){
        check_item(item);
        Node oldLast = last;
        last = new Node();
        last.item = item;
        last.next = null;
        last.prev = oldLast;
        if (isEmpty()){
            first = last;
        }
        else{
            oldLast.next = last;
        }
        length ++;
    }

    public Item removeFirst() {
        if (isEmpty()){
           throw new java.util.NoSuchElementException("No element in deque !");
        }
        Node oldFirst = first ;
        first = first.next;
        oldFirst.next = null;
        length -- ;
        if (isEmpty()){
            last = first;
        }
        else{
            first.prev = null;
        }
        return oldFirst.item;
    }

    public Item removeLast(){
        if (isEmpty()){
            throw new java.util.NoSuchElementException("No element in deque !");
        }
        Node oldLast = last;
        last = last.prev;
        oldLast.prev = null;
        length -- ;
        if (isEmpty()){
            first = last;
        }
        else {
            last.next =null;
        }
        return oldLast.item;
    }

    public Iterator<Item> iterator(){
        return new DequeIterator(first);
    }

    private class DequeIterator implements Iterator<Item> {
        private Node current;
        public DequeIterator(Node first){
            current = first;
        }
        
        public boolean hasNext(){
            return current != null;
        }

        public Item next(){
            if(!hasNext()){
                throw new java.util.NoSuchElementException("Is Empty!");
            }
            Item ret = current.item;
            current = current.next;
            return ret;
        }

        public void remove(){
            throw new UnsupportedOperationException("Unsupported Operation");
        }
    }

    public static void main(String[] args){
        Deque<Integer> deque = new Deque<>();
        for (int i =0 ; i<4; i ++){
            deque.addFirst(i);
            deque.addLast(i + 4);
        }
        for (int i : deque){
            StdOut.print(i + " ");
        }
        StdOut.print("\n");
        StdOut.println(deque.size());
        while (!deque.isEmpty()){
            StdOut.println(deque.removeLast());
            StdOut.println(deque.removeFirst());
        }
    }

}