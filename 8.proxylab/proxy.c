#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <signal.h>
#include "csapp.h"

/* Recommended max cache and object sizes */
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400

/* You won't lose style points for including this long line in your code */
static const char *user_agent_hdr = "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3\r\n";


void *thread(void *vargp);
void proxy(int clientfd);
void parse_uri(char *uri, char *hostname, char *path, int *port);
void build_request_header(char *request_header, char *hostname, char *path, rio_t *client_rio);

int main(int argc, char **argv)
{
    int listenfd, *connfd;
    socklen_t clientlen;
    struct sockaddr_storage clientaddr;
    pthread_t tid;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <port>\n", argv[0]);
        exit(1);
    }

    listenfd = Open_listenfd(argv[1]);
    while (1) {
        clientlen = sizeof(clientaddr);
        connfd = Malloc(sizeof(int));
        *connfd = Accept(listenfd, (SA *)&clientaddr, &clientlen);
        Pthread_create(&tid, NULL, thread, connfd);
    }
}

void *thread(void *vargp)
{
    int connfd = *((int *)vargp);
    Pthread_detach(pthread_self());
    Free(vargp);
    proxy(connfd);
    Close(connfd);
    return NULL;
}

void proxy(int clientfd)
{
    char buf[MAXLINE], method[MAXLINE], uri[MAXLINE], version[MAXLINE];
    char hostname[MAXLINE], path[MAXLINE], request_header[MAXLINE];
    char port_str[10];  
    int port, serverfd;
    rio_t rio, server_rio;

    Rio_readinitb(&rio, clientfd);
    Rio_readlineb(&rio, buf, MAXLINE);
    sscanf(buf, "%s %s %s", method, uri, version);

    parse_uri(uri, hostname, path, &port);

    sprintf(port_str, "%d", port);

    build_request_header(request_header, hostname, path, &rio);

    serverfd = Open_clientfd(hostname, port_str);
    Rio_readinitb(&server_rio, serverfd);

    /* Forward request to server */
    Rio_writen(serverfd, request_header, strlen(request_header));

    /* Forward response to client */
    size_t n;
    while ((n = Rio_readlineb(&server_rio, buf, MAXLINE)) != 0) {
        Rio_writen(clientfd, buf, n);
    }

    Close(serverfd);
}

void parse_uri(char *uri, char *hostname, char *path, int *port)
{
    *port = 80;
    char *ptr = strstr(uri, "//");
    ptr = ptr ? ptr + 2 : uri;

    char *host_end = strchr(ptr, ':');
    if (host_end) {
        *host_end = '\0';
        strcpy(hostname, ptr);
        sscanf(host_end + 1, "%d%s", port, path);
    } else {
        host_end = strchr(ptr, '/');
        if (host_end) {
            *host_end = '\0';
            strcpy(hostname, ptr);
            *host_end = '/';
            strcpy(path, host_end);
        } else {
            strcpy(hostname, ptr);
            strcpy(path, "");
        }
    }
}

void build_request_header(char *request_header, char *hostname, char *path, rio_t *client_rio)
{
    char buf[MAXLINE], host_header[MAXLINE], other_header[MAXLINE];
    
    sprintf(host_header, "Host: %s\r\n", hostname);
    strcpy(other_header, "");

    while (Rio_readlineb(client_rio, buf, MAXLINE) > 0) {
        if (strcmp(buf, "\r\n") == 0) break;
        if (!strncasecmp(buf, "Host:", 5)) continue;
        if (!strncasecmp(buf, "User-Agent:", 11)) continue;
        if (!strncasecmp(buf, "Connection:", 11)) continue;
        if (!strncasecmp(buf, "Proxy-Connection:", 17)) continue;
        strcat(other_header, buf);
    }

    sprintf(request_header,
            "GET %s HTTP/1.0\r\n"
            "%s"
            "%s"
            "%s"
            "Connection: close\r\n"
            "Proxy-Connection: close\r\n\r\n",
            path, host_header, user_agent_hdr, other_header);
}
