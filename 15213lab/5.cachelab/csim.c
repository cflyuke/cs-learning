#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "cachelab.h"

typedef struct {
    int valid;
    unsigned long tag;
    int timestamp;
} CacheLine;

typedef struct {
    CacheLine *lines;
} CacheSet;

typedef struct {
    int s;
    int E;
    int b;
    int S;
    int B;
    CacheSet *sets;
} Cache;

Cache cache;
int hits = 0, misses = 0, evictions = 0;
int time = 0;

void initCache(int s, int E, int b) {
    cache.s = s;
    cache.E = E;
    cache.b = b;
    cache.S = 1 << s;
    cache.B = 1 << b;
    cache.sets = (CacheSet *)malloc(cache.S * sizeof(CacheSet));
    for (int i = 0; i < cache.S; i++) {
        cache.sets[i].lines = (CacheLine *)malloc(E * sizeof(CacheLine));
        for (int j = 0; j < E; j++) {
            cache.sets[i].lines[j].valid = 0;
            cache.sets[i].lines[j].timestamp = 0;
        }
    }
}

void accessCache(unsigned long address) {
    int setIndex = (address >> cache.b) & ((1 << cache.s) - 1);
    unsigned long tag = address >> (cache.s + cache.b);
    CacheSet set = cache.sets[setIndex];
    int hit = 0, emptyIndex = -1, lruIndex = 0, minTime = time;

    for (int i = 0; i < cache.E; i++) {
        if (set.lines[i].valid && set.lines[i].tag == tag) {
            hits++;
            set.lines[i].timestamp = time++;
            hit = 1;
            break;
        }
        if (!set.lines[i].valid && emptyIndex == -1) {
            emptyIndex = i;
        }
        if (set.lines[i].timestamp < minTime) {
            minTime = set.lines[i].timestamp;
            lruIndex = i;
        }
    }

    if (!hit) {
        misses++;
        if (emptyIndex != -1) {
            set.lines[emptyIndex].valid = 1;
            set.lines[emptyIndex].tag = tag;
            set.lines[emptyIndex].timestamp = time++;
        } else {
            evictions++;
            set.lines[lruIndex].tag = tag;
            set.lines[lruIndex].timestamp = time++;
        }
    }
}

void freeCache() {
    for (int i = 0; i < cache.S; i++) {
        free(cache.sets[i].lines);
    }
    free(cache.sets);
}

int main(int argc, char **argv) {
    int s, E, b;
    char *tracefile;
    char c;
    while ((c = getopt(argc, argv, "s:E:b:t:")) != -1) {
        switch (c) {
            case 's':
                s = atoi(optarg);
                break;
            case 'E':
                E = atoi(optarg);
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 't':
                tracefile = optarg;
                break;
            default:
                exit(1);
        }
    }

    initCache(s, E, b);

    FILE *file = fopen(tracefile, "r");
    if (!file) {
        perror("Error opening trace file");
        exit(1);
    }

    char operation;
    unsigned long address;
    int size;
    while (fscanf(file, " %c %lx,%d", &operation, &address, &size) == 3) {
        if (operation == 'L' || operation == 'S' || operation == 'M') {
            accessCache(address);
            if (operation == 'M') {
                accessCache(address);
            }
        }
    }

    fclose(file);
    printSummary(hits, misses, evictions);
    freeCache();
    return 0;
}