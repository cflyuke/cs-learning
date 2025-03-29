#include "csapp.h"

int main(int argc, char** argv){
    if(argc != 2){
        unix_error("format ./myhostinfo www.website.com");
    }
    struct addrinfo hints;
    struct addrinfo* res, *cur;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    Getaddrinfo(argv[1], NULL, &hints, &res);
    char hostName[MAXLINE];
    int flags = NI_NUMERICHOST;
    for (cur = res; cur != NULL ; cur = cur->ai_next){
        Getnameinfo(cur->ai_addr, cur->ai_addrlen, hostName, MAXLINE, NULL, 0, flags);
        printf("canonname: %s, hostname: %s\n", cur->ai_canonname, hostName);
    }
    Freeaddrinfo(res);
    exit(EXIT_SUCCESS);
}