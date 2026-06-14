# Chapter 23: 套接字编程与网络应用开发

> **学习目标**：
> - 理解套接字 API 的基本概念与系统调用
> - 掌握 TCP 客户端/服务器编程：socket/bind/listen/accept/connect/read/write/close
> - 掌握 UDP 套接字编程与无连接通信
> - 理解并发服务器模型：多进程、多线程、I/O 多路复用
> - 掌握 epoll 事件通知器的红黑树与就绪链表设计
> - 理解事件驱动架构与高性能服务器设计（Reactor/Proactor）
> - 了解原始套接字与 libpcap 数据包捕获

---

## 23.1 套接字 API 概述

### 23.1.1 什么是套接字

套接字（Socket）是网络编程的基本接口，它是网络通信的端点：

```
套接字的抽象:
  ┌─────────────────────────────────────────────────┐
  │                  应用程序                        │
  │    write()    ┌─────────┐    read()             │
  │    ────────►  │  套接字  │  ◄────────            │
  │               │ (Socket) │                      │
  │               └────┬────┘                       │
  └────────────────────┼───────────────────────────┘
                       │
  ┌────────────────────┼───────────────────────────┐
  │               内核空间                          │
  │               ┌────┴────┐                       │
  │               │ 套接字层 │                       │
  │               └────┬────┘                       │
  │               ┌────┴────┐                       │
  │               │ 协议栈   │                       │
  │               └────┬────┘                       │
  │               ┌────┴────┐                       │
  │               │ 网络设备 │                       │
  │               └─────────┘                       │
  └─────────────────────────────────────────────────┘

套接字类型:
  1. 流套接字 (SOCK_STREAM): TCP，可靠、有序、字节流
  2. 数据报套接字 (SOCK_DGRAM): UDP，无连接、消息边界
  3. 原始套接字 (SOCK_RAW): 直接访问 IP 层
```

### 23.1.2 套接字层在内核中的实现架构

```
文件描述符 → 套接字 → 协议栈的映射:

  用户空间:
    fd = socket(AF_INET, SOCK_STREAM, 0)
    → 返回文件描述符 3

  内核空间:
    进程文件描述符表:
      [0] → stdin
      [1] → stdout
      [2] → stderr
      [3] → struct file → struct socket → struct sock → TCP 协议栈
      [4] → struct file → struct socket → struct sock → UDP 协议栈

  关键数据结构:
    struct file:    文件抽象，提供 read/write 接口
    struct socket:  套接字抽象，包含协议族、类型
    struct sock:    传输层端点，包含连接状态、缓冲区
    struct tcp_sock: TCP 特有的状态（序列号、窗口等）
```

```c
// 内核中的套接字结构（简化）
struct socket {
    socket_state        state;      // 连接状态
    short               type;       // 套接字类型
    unsigned long       flags;
    struct file         *file;      // 关联的文件
    struct sock         *sk;        // 传输层端点
    const struct proto_ops *ops;    // 操作函数表
};

struct sock {
    struct sock_common  __sk_common;
    unsigned int        sk_padding;
    atomic_t            sk_rmem_alloc;  // 接收缓冲区使用量
    atomic_t            sk_wmem_alloc;  // 发送缓冲区使用量
    struct sk_buff_head sk_receive_queue; // 接收队列
    struct sk_buff_head sk_write_queue;   // 发送队队列
    int                 sk_rcvbuf;        // 接收缓冲区大小
    int                 sk_sndbuf;        // 发送缓冲区大小
};
```

<div data-component="SocketLayerDiagram"></div>

---

## 23.2 TCP 客户端/服务器编程

### 23.2.1 TCP 服务器端流程

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];

    // 1. 创建套接字
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 设置 SO_REUSEADDR，允许重用端口
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 2. 绑定地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  // 监听所有接口
    server_addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 3. 监听连接
    if (listen(server_fd, 128) < 0) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);

    // 4. 接受连接
    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept failed");
            continue;
        }

        printf("Client connected: %s:%d\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port));

        // 5. 读写数据
        ssize_t bytes_read;
        while ((bytes_read = read(client_fd, buffer, BUFFER_SIZE)) > 0) {
            printf("Received %zd bytes: %.*s\n", bytes_read, (int)bytes_read, buffer);
            // 回显数据
            write(client_fd, buffer, bytes_read);
        }

        // 6. 关闭连接
        close(client_fd);
        printf("Client disconnected\n");
    }

    close(server_fd);
    return 0;
}
```

### 23.2.2 TCP 客户端流程

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP "127.0.0.1"
#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    char *message = "Hello, Server!";

    // 1. 创建套接字
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 2. 设置服务器地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    // 3. 连接服务器
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        exit(EXIT_FAILURE);
    }

    printf("Connected to server\n");

    // 4. 发送数据
    write(sockfd, message, strlen(message));
    printf("Sent: %s\n", message);

    // 5. 接收响应
    ssize_t bytes_read = read(sockfd, buffer, BUFFER_SIZE);
    if (bytes_read > 0) {
        printf("Received: %.*s\n", (int)bytes_read, buffer);
    }

    // 6. 关闭连接
    close(sockfd);
    return 0;
}
```

### 23.2.3 TCP 状态转换图

```
TCP 三次握手:
  客户端:                    服务器:
    CLOSED                   LISTEN
      │                        │
      │ connect()              │
      ▼                        │
    SYN_SENT ── SYN ──────►   │
      │                        ▼
      │                  SYN_RCVD ◄── SYN
      │                        │
      ◄── SYN+ACK ───────────┘
      │                        │
    ESTABLISHED                │
      │                        │
      ◄── ACK ────────────────►
                                │
                              ESTABLISHED

TCP 四次挥手:
  客户端:                    服务器:
    ESTABLISHED              ESTABLISHED
      │                        │
      │ close()                │
      ▼                        │
    FIN_WAIT_1 ── FIN ────►   │
      │                        ▼
      │                  CLOSE_WAIT ◄── FIN
      │                        │
      ◄── ACK ────────────────┘
      │                        │
    FIN_WAIT_2                │
      │                   (应用调用 close())
      │                        │
      │                        ▼
      ◄── FIN ──────────── LAST_ACK
      │                        │
    TIME_WAIT                  │
      │                        │
      ▼                        │
    CLOSED                CLOSED ◄── ACK
```

<div data-component="TCPFlowDiagram"></div>

---

## 23.3 UDP 套接字编程

### 23.3.1 UDP 服务器

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];

    // 1. 创建 UDP 套接字
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 2. 绑定地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    printf("UDP server listening on port %d\n", PORT);

    // 3. 收发数据
    while (1) {
        ssize_t n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                            (struct sockaddr *)&client_addr, &client_len);
        if (n < 0) {
            perror("recvfrom failed");
            continue;
        }

        printf("Received from %s:%d: %.*s\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port),
               (int)n, buffer);

        // 回显数据
        sendto(sockfd, buffer, n, 0,
               (struct sockaddr *)&client_addr, client_len);
    }

    close(sockfd);
    return 0;
}
```

### 23.3.2 UDP 客户端

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP "127.0.0.1"
#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    char *message = "Hello, UDP Server!";

    // 1. 创建 UDP 套接字
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 2. 设置服务器地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    // 3. 发送数据
    sendto(sockfd, message, strlen(message), 0,
           (struct sockaddr *)&server_addr, sizeof(server_addr));
    printf("Sent: %s\n", message);

    // 4. 接收响应
    socklen_t addr_len = sizeof(server_addr);
    ssize_t n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                        (struct sockaddr *)&server_addr, &addr_len);
    if (n > 0) {
        printf("Received: %.*s\n", (int)n, buffer);
    }

    close(sockfd);
    return 0;
}
```

### 23.3.3 TCP vs UDP 套接字编程对比

```
┌──────────────────────┬──────────────────┬──────────────────┐
│       操作           │      TCP         │      UDP         │
├──────────────────────┼──────────────────┼──────────────────┤
│ 创建                 │ SOCK_STREAM      │ SOCK_DGRAM       │
│ 绑定                 │ bind()           │ bind()           │
│ 监听                 │ listen()         │ 不需要           │
│ 接受连接             │ accept()         │ 不需要           │
│ 建立连接             │ connect()        │ 可选             │
│ 发送                 │ write/send       │ sendto()         │
│ 接收                 │ read/recv        │ recvfrom()       │
│ 关联地址             │ 连接时自动确定   │ 每次 sendto 指定 │
│ 消息边界             │ 无               │ 有               │
│ 可靠性               │ 可靠             │ 不可靠           │
└──────────────────────┴──────────────────┴──────────────────┘
```

<div data-component="UDPDemo"></div>

---

## 23.4 并发服务器

### 23.4.1 多进程服务器

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>

#define PORT 8080
#define BUFFER_SIZE 1024

void handle_client(int client_fd) {
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = read(client_fd, buffer, BUFFER_SIZE)) > 0) {
        write(client_fd, buffer, bytes_read);
    }

    close(client_fd);
    exit(0);
}

void sigchld_handler(int sig) {
    // 回收子进程，避免僵尸进程
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

int main() {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    // 设置 SIGCHLD 处理器
    signal(SIGCHLD, sigchld_handler);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(server_fd, 128);

    printf("Multi-process server listening on port %d\n", PORT);

    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept failed");
            continue;
        }

        // 创建子进程处理客户端
        pid_t pid = fork();
        if (pid == 0) {
            // 子进程
            close(server_fd);  // 关闭监听套接字
            handle_client(client_fd);
        } else if (pid > 0) {
            // 父进程
            close(client_fd);  // 关闭客户端套接字
        } else {
            perror("fork failed");
            close(client_fd);
        }
    }

    close(server_fd);
    return 0;
}
```

### 23.4.2 多线程服务器

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

#define PORT 8080
#define BUFFER_SIZE 1024

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
} client_info_t;

void *handle_client(void *arg) {
    client_info_t *info = (client_info_t *)arg;
    int client_fd = info->client_fd;
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read;

    // 分离线程，自动回收资源
    pthread_detach(pthread_self());

    printf("Thread %lu handling client %s:%d\n",
           pthread_self(),
           inet_ntoa(info->client_addr.sin_addr),
           ntohs(info->client_addr.sin_port));

    while ((bytes_read = read(client_fd, buffer, BUFFER_SIZE)) > 0) {
        write(client_fd, buffer, bytes_read);
    }

    close(client_fd);
    free(info);
    return NULL;
}

int main() {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    pthread_t thread_id;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(server_fd, 128);

    printf("Multi-thread server listening on port %d\n", PORT);

    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept failed");
            continue;
        }

        client_info_t *info = malloc(sizeof(client_info_t));
        info->client_fd = client_fd;
        info->client_addr = client_addr;

        pthread_create(&thread_id, NULL, handle_client, info);
    }

    close(server_fd);
    return 0;
}
```

<div data-component="ConcurrentServerDemo"></div>

---

## 23.5 I/O 多路复用

### 23.5.1 select

```c
#include <sys/select.h>

// select 最大监听文件描述符数量
#define MAX_FD 1024

int main() {
    fd_set read_fds;
    int max_fd;
    int client_fds[MAX_FD] = {0};

    // 初始化
    FD_ZERO(&read_fds);
    FD_SET(server_fd, &read_fds);
    max_fd = server_fd;

    while (1) {
        // select 会修改 fd_set，需要每次重新设置
        fd_set temp_fds = read_fds;

        int ready = select(max_fd + 1, &temp_fds, NULL, NULL, NULL);
        if (ready < 0) {
            perror("select failed");
            continue;
        }

        // 检查新连接
        if (FD_ISSET(server_fd, &temp_fds)) {
            int client_fd = accept(server_fd, ...);
            FD_SET(client_fd, &read_fds);
            if (client_fd > max_fd) max_fd = client_fd;
        }

        // 检查已连接的客户端
        for (int i = 0; i <= max_fd; i++) {
            if (i != server_fd && FD_ISSET(i, &temp_fds)) {
                char buffer[1024];
                ssize_t n = read(i, buffer, sizeof(buffer));
                if (n <= 0) {
                    close(i);
                    FD_CLR(i, &read_fds);
                } else {
                    write(i, buffer, n);
                }
            }
        }
    }
}
```

**select 的局限**：

```
select 的问题:
  1. FD_SETSIZE 限制（通常 1024）
  2. 每次调用需要重新设置 fd_set
  3. 每次调用需要遍历所有文件描述符
  4. 用户空间和内核空间之间复制 fd_set
  5. 时间复杂度 O(n)，n 为最大文件描述符
```

### 23.5.2 poll

```c
#include <poll.h>

#define MAX_CLIENTS 10000

int main() {
    struct pollfd fds[MAX_CLIENTS];
    int nfds = 1;

    // 初始化
    fds[0].fd = server_fd;
    fds[0].events = POLLIN;

    while (1) {
        int ready = poll(fds, nfds, -1);
        if (ready < 0) {
            perror("poll failed");
            continue;
        }

        // 检查新连接
        if (fds[0].revents & POLLIN) {
            int client_fd = accept(server_fd, ...);
            fds[nfds].fd = client_fd;
            fds[nfds].events = POLLIN;
            nfds++;
        }

        // 检查已连接的客户端
        for (int i = 1; i < nfds; i++) {
            if (fds[i].revents & POLLIN) {
                char buffer[1024];
                ssize_t n = read(fds[i].fd, buffer, sizeof(buffer));
                if (n <= 0) {
                    close(fds[i].fd);
                    // 移除：将最后一个移到当前位置
                    fds[i] = fds[nfds - 1];
                    nfds--;
                    i--;
                } else {
                    write(fds[i].fd, buffer, n);
                }
            }
        }
    }
}
```

**poll 的改进与局限**：

```
poll 相比 select 的改进:
  1. 没有 FD_SETSIZE 限制
  2. 使用 pollfd 结构数组，更灵活
  3. events 和 revents 分离，无需每次重新设置

poll 仍然存在的问题:
  1. 每次调用仍需遍历所有文件描述符 O(n)
  2. 用户空间和内核空间之间复制 pollfd 数组
  3. 性能随连接数增加而下降
```

### 23.5.3 epoll 事件通知器的红黑树与就绪链表设计

epoll 是 Linux 特有的高性能 I/O 多路复用机制：

```
epoll 内部数据结构:

  ┌─────────────────────────────────────────────────────────┐
  │                    epoll 实例                           │
  │                                                         │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │              红黑树 (RB-Tree)                     │  │
  │  │          存储所有被监控的文件描述符               │  │
  │  │                                                  │  │
  │  │         ┌─────┐                                  │  │
  │  │         │ fd5 │                                  │  │
  │  │        /     \                                   │  │
  │  │    ┌───┐     ┌───┐                              │  │
  │  │    │fd3│     │fd7│                              │  │
  │  │   /   \     /   \                               │  │
  │  │ ┌───┐ ┌───┐ ┌───┐ ┌───┐                        │  │
  │  │ │fd1│ │fd4│ │fd6│ │fd8│                        │  │
  │  │ └───┘ └───┘ └───┘ └───┘                        │  │
  │  └──────────────────────────────────────────────────┘  │
  │                                                         │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │            就绪链表 (Ready List)                  │  │
  │  │        存储有事件发生的文件描述符                 │  │
  │  │                                                  │  │
  │  │    fd3 → fd7 → fd1 → NULL                       │  │
  │  └──────────────────────────────────────────────────┘  │
  │                                                         │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │            回调函数注册                           │  │
  │  │  当文件描述符有事件时，回调函数将其加入就绪链表  │  │
  │  └──────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘

epoll 的优势:
  1. 红黑树查找: O(log n) 插入/删除
  2. 就绪链表: O(1) 获取就绪事件
  3. 无需每次复制文件描述符集合
  4. 回调机制: 只返回有事件的文件描述符
  5. 性能与连接总数无关，只与活跃连接数相关
```

```c
#include <sys/epoll.h>

#define MAX_EVENTS 10000
#define TIMEOUT 5000  // 5 秒

int main() {
    int epoll_fd, nfds;
    struct epoll_event event, events[MAX_EVENTS];

    // 1. 创建 epoll 实例
    epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) {
        perror("epoll_create1 failed");
        exit(EXIT_FAILURE);
    }

    // 2. 添加监听套接字到 epoll
    event.events = EPOLLIN;  // 监听可读事件
    event.data.fd = server_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event) < 0) {
        perror("epoll_ctl failed");
        exit(EXIT_FAILURE);
    }

    printf("epoll server listening on port %d\n", PORT);

    while (1) {
        // 3. 等待事件
        nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, TIMEOUT);
        if (nfds < 0) {
            perror("epoll_wait failed");
            continue;
        }

        // 4. 处理事件
        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == server_fd) {
                // 新连接
                int client_fd = accept(server_fd, ...);
                // 设置非阻塞
                fcntl(client_fd, F_SETFL, O_NONBLOCK);
                // 添加到 epoll
                event.events = EPOLLIN | EPOLLET;  // 边缘触发
                event.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &event);
            } else {
                // 已连接客户端的数据
                int client_fd = events[i].data.fd;
                char buffer[4096];
                ssize_t n;

                // 边缘触发模式下，需要读完所有数据
                while ((n = read(client_fd, buffer, sizeof(buffer))) > 0) {
                    write(client_fd, buffer, n);
                }

                if (n == 0 || (n < 0 && errno != EAGAIN)) {
                    // 连接关闭或错误
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client_fd, NULL);
                    close(client_fd);
                }
            }
        }
    }

    close(epoll_fd);
    return 0;
}
```

**epoll 触发模式**：

```
水平触发 (Level Triggered, LT):
  - 默认模式
  - 只要文件描述符处于就绪状态，就会持续通知
  - 编程简单，但可能有重复通知

边缘触发 (Edge Triggered, ET):
  - 只在状态变化时通知一次
  - 必须一次性读完所有数据
  - 性能更高，但编程更复杂
  - 必须使用非阻塞 I/O

示例:
  LT 模式: 缓冲区有数据 → 每次 epoll_wait 都返回
  ET 模式: 缓冲区有新数据 → 只返回一次
```

<div data-component="EpollDemo"></div>

---

## 23.6 非阻塞 I/O 与事件驱动架构

### 23.6.1 非阻塞 I/O

```c
#include <fcntl.h>

// 设置非阻塞模式
int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// 非阻塞读
ssize_t nonblocking_read(int fd, void *buf, size_t count) {
    ssize_t n = read(fd, buf, count);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 没有数据可读，稍后再试
            return 0;
        }
        // 真正的错误
        return -1;
    }
    return n;
}

// 非阻塞写
ssize_t nonblocking_write(int fd, const void *buf, size_t count) {
    ssize_t n = write(fd, buf, count);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 发送缓冲区满，稍后再试
            return 0;
        }
        return -1;
    }
    return n;
}
```

### 23.6.2 Reactor 模式

```
Reactor 模式架构:

  ┌─────────────────────────────────────────────────────────┐
  │                    Reactor                               │
  │                                                         │
  │  ┌─────────────┐    ┌─────────────┐                    │
  │  │ 事件分离器   │    │ 事件处理器   │                    │
  │  │ (epoll)     │───►│ (Handler)   │                    │
  │  └─────────────┘    └─────────────┘                    │
  │         ▲                  │                            │
  │         │                  ▼                            │
  │  ┌─────────────┐    ┌─────────────┐                    │
  │  │   I/O 操作   │◄──│ 业务逻辑    │                    │
  │  │             │    │             │                    │
  │  └─────────────┘    └─────────────┘                    │
  └─────────────────────────────────────────────────────────┘

Reactor 工作流程:
  1. 注册感兴趣的事件（连接、读、写）
  2. 事件分离器等待事件发生
  3. 事件发生时，调用对应的事件处理器
  4. 事件处理器执行业务逻辑
  5. 返回事件分离器继续等待
```

```python
import selectors
import socket

class Reactor:
    """Reactor 模式实现"""
    def __init__(self):
        self.selector = selectors.DefaultSelector()
        self.handlers = {}

    def register(self, sock, event, handler):
        """注册事件处理器"""
        self.selector.register(sock, event, data=handler)
        self.handlers[sock.fileno()] = handler

    def unregister(self, sock):
        """注销事件处理器"""
        self.selector.unregister(sock)
        if sock.fileno() in self.handlers:
            del self.handlers[sock.fileno()]

    def run(self):
        """事件循环"""
        while True:
            events = self.selector.select(timeout=-1)
            for key, mask in events:
                handler = key.data
                if handler:
                    handler.handle_event(mask)

class AcceptHandler:
    """连接接受处理器"""
    def __init__(self, reactor, server_sock):
        self.reactor = reactor
        self.server_sock = server_sock

    def handle_event(self, mask):
        if mask & selectors.EVENT_READ:
            client_sock, addr = self.server_sock.accept()
            print(f"New connection from {addr}")
            client_sock.setblocking(False)
            # 注册客户端处理器
            handler = ClientHandler(client_sock, self.reactor)
            self.reactor.register(client_sock, selectors.EVENT_READ, handler)

class ClientHandler:
    """客户端数据处理器"""
    def __init__(self, sock, reactor):
        self.sock = sock
        self.reactor = reactor
        self.buffer = b''

    def handle_event(self, mask):
        if mask & selectors.EVENT_READ:
            data = self.sock.recv(4096)
            if data:
                self.buffer += data
                # 注册写事件
                self.reactor.register(self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, self)
            else:
                # 连接关闭
                self.reactor.unregister(self.sock)
                self.sock.close()

        if mask & selectors.EVENT_WRITE:
            if self.buffer:
                sent = self.sock.send(self.buffer)
                self.buffer = self.buffer[sent:]
                if not self.buffer:
                    # 写完，只监听读事件
                    self.reactor.register(self.sock, selectors.EVENT_READ, self)
```

<div data-component="ReactorDemo"></div>

### 23.6.3 Proactor 模式

```
Proactor 模式架构:

  ┌─────────────────────────────────────────────────────────┐
  │                    Proactor                              │
  │                                                         │
  │  ┌─────────────┐    ┌─────────────┐                    │
  │  │ 异步 I/O     │    │ 完成处理器   │                    │
  │  │ (aio_read)  │───►│ (Completion)│                    │
  │  └─────────────┘    └─────────────┘                    │
  │         ▲                  │                            │
  │         │                  ▼                            │
  │  ┌─────────────┐    ┌─────────────┐                    │
  │  │   内核       │    │ 业务逻辑    │                    │
  │  │ (io_uring)  │    │             │                    │
  │  └─────────────┘    └─────────────┘                    │
  └─────────────────────────────────────────────────────────┘

Reactor vs Proactor:
  Reactor:  应用程序主动检查 I/O 就绪状态（同步 I/O 多路复用）
  Proactor: 内核完成 I/O 操作后通知应用程序（异步 I/O）
```

---

## 23.7 服务器连接管理器的连接池与超时处理

### 23.7.1 连接池管理

```python
import time
import socket
from collections import OrderedDict
from typing import Optional
from dataclasses import dataclass

@dataclass
class Connection:
    sock: socket.socket
    created_at: float
    last_used: float
    request_count: int = 0

class ConnectionPool:
    """连接池管理器"""
    def __init__(self, max_connections: int = 10000, timeout: float = 60.0):
        self.max_connections = max_connections
        self.timeout = timeout  # 空闲超时（秒）
        self.connections: OrderedDict[int, Connection] = {}  # fd -> Connection
        self.total_created = 0
        self.total_closed = 0

    def add_connection(self, sock: socket.socket) -> int:
        """添加新连接"""
        if len(self.connections) >= self.max_connections:
            # 淘汰最久未使用的连接
            self._evict_oldest()

        fd = sock.fileno()
        now = time.time()
        self.connections[fd] = Connection(
            sock=sock,
            created_at=now,
            last_used=now,
        )
        self.total_created += 1
        return fd

    def get_connection(self, fd: int) -> Optional[Connection]:
        """获取连接"""
        conn = self.connections.get(fd)
        if conn:
            conn.last_used = time.time()
            conn.request_count += 1
        return conn

    def remove_connection(self, fd: int):
        """移除连接"""
        if fd in self.connections:
            conn = self.connections.pop(fd)
            try:
                conn.sock.close()
            except:
                pass
            self.total_closed += 1

    def cleanup_expired(self):
        """清理过期连接"""
        now = time.time()
        expired_fds = []

        for fd, conn in self.connections.items():
            if now - conn.last_used > self.timeout:
                expired_fds.append(fd)

        for fd in expired_fds:
            self.remove_connection(fd)

        return len(expired_fds)

    def _evict_oldest(self):
        """淘汰最久未使用的连接"""
        if self.connections:
            fd = next(iter(self.connections))
            self.remove_connection(fd)

    def get_stats(self) -> dict:
        """获取连接池统计信息"""
        return {
            'active_connections': len(self.connections),
            'total_created': self.total_created,
            'total_closed': self.total_closed,
            'max_connections': self.max_connections,
        }


class TimeoutManager:
    """超时管理器"""
    def __init__(self):
        self.timers = {}  # {fd: (callback, timeout, start_time)}

    def set_timeout(self, fd: int, timeout: float, callback):
        """设置超时回调"""
        self.timers[fd] = (callback, timeout, time.time())

    def cancel_timeout(self, fd: int):
        """取消超时"""
        if fd in self.timers:
            del self.timers[fd]

    def check_timeouts(self):
        """检查超时的连接"""
        now = time.time()
        expired = []

        for fd, (callback, timeout, start_time) in self.timers.items():
            if now - start_time > timeout:
                expired.append(fd)

        for fd in expired:
            callback, _, _ = self.timers.pop(fd)
            callback(fd)

        return len(expired)
```

<div data-component="ConnectionPoolDemo"></div>

---

## 23.8 高性能服务器设计

### 23.8.1 线程池模型

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class ThreadPoolServer:
    """线程池服务器"""
    def __init__(self, max_workers: int = 100, max_queue_size: int = 10000):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False

    def start(self, handler, server_socket):
        """启动服务器"""
        self.running = True

        # 接受连接的线程
        accept_thread = threading.Thread(target=self._accept_loop,
                                         args=(server_socket, handler))
        accept_thread.daemon = True
        accept_thread.start()

    def _accept_loop(self, server_socket, handler):
        """接受连接循环"""
        while self.running:
            try:
                client_socket, addr = server_socket.accept()
                # 提交到线程池处理
                self.executor.submit(handler, client_socket, addr)
            except Exception as e:
                print(f"Accept error: {e}")

    def stop(self):
        """停止服务器"""
        self.running = False
        self.executor.shutdown(wait=True)
```

### 23.8.2 单线程事件循环 + 非阻塞 I/O

```python
import selectors
import socket

class EventLoopServer:
    """单线程事件循环服务器"""
    def __init__(self):
        self.selector = selectors.DefaultSelector()
        self.connections = {}

    def start(self, host: str, port: int):
        """启动服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setblocking(False)
        server_socket.bind((host, port))
        server_socket.listen(10000)

        self.selector.register(server_socket, selectors.EVENT_READ, data='accept')

        print(f"Event loop server listening on {host}:{port}")

        while True:
            events = self.selector.select(timeout=-1)
            for key, mask in events:
                if key.data == 'accept':
                    self._accept_connection(key.fileobj)
                else:
                    self._handle_client(key, mask)

    def _accept_connection(self, server_socket):
        """接受新连接"""
        client_socket, addr = server_socket.accept()
        client_socket.setblocking(False)
        self.selector.register(client_socket, selectors.EVENT_READ, data='client')
        self.connections[client_socket.fileno()] = {
            'socket': client_socket,
            'address': addr,
            'buffer': b'',
        }

    def _handle_client(self, key, mask):
        """处理客户端数据"""
        client_socket = key.fileobj
        fd = client_socket.fileno()

        if mask & selectors.EVENT_READ:
            try:
                data = client_socket.recv(4096)
                if data:
                    self.connections[fd]['buffer'] += data
                    # 注册写事件
                    self.selector.modify(client_socket, selectors.EVENT_READ | selectors.EVENT_WRITE)
                else:
                    self._close_connection(fd)
            except ConnectionResetError:
                self._close_connection(fd)

        if mask & selectors.EVENT_WRITE:
            buffer = self.connections[fd]['buffer']
            if buffer:
                try:
                    sent = client_socket.send(buffer)
                    self.connections[fd]['buffer'] = buffer[sent:]
                    if not self.connections[fd]['buffer']:
                        # 写完，只监听读事件
                        self.selector.modify(client_socket, selectors.EVENT_READ)
                except:
                    self._close_connection(fd)

    def _close_connection(self, fd: int):
        """关闭连接"""
        if fd in self.connections:
            conn = self.connections.pop(fd)
            self.selector.unregister(conn['socket'])
            conn['socket'].close()
```

<div data-component="HighPerfServerDemo"></div>

---

## 23.9 原始套接字与数据包捕获

### 23.9.1 原始套接字

```c
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>

// 创建原始套接字（需要 root 权限）
int raw_sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);

// 设置 IP_HDRINCL，自己构造 IP 头部
int opt = 1;
setsockopt(raw_sock, IPPROTO_IP, IP_HDRINCL, &opt, sizeof(opt));

// 接收所有数据包
while (1) {
    char buffer[65536];
    struct sockaddr_in source;
    socklen_t source_len = sizeof(source);

    ssize_t n = recvfrom(raw_sock, buffer, sizeof(buffer), 0,
                        (struct sockaddr *)&source, &source_len);

    // 解析 IP 头部
    struct iphdr *ip_header = (struct iphdr *)buffer;
    printf("Source IP: %s\n", inet_ntoa(*(struct in_addr *)&ip_header->saddr));
    printf("Dest IP: %s\n", inet_ntoa(*(struct in_addr *)&ip_header->daddr));

    // 如果是 TCP，解析 TCP 头部
    if (ip_header->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp_header = (struct tcphdr *)(buffer + ip_header->ihl * 4);
        printf("Source Port: %d\n", ntohs(tcp_header->source));
        printf("Dest Port: %d\n", ntohs(tcp_header->dest));
    }
}
```

### 23.9.2 libpcap 数据包捕获引擎与 BPF 过滤器实现

```c
#include <pcap/pcap.h>

// BPF 过滤器示例
// 只捕获目标端口为 80 的 TCP 数据包
const char *filter_exp = "tcp port 80";
struct bpf_program fp;

// 编译过滤器
if (pcap_compile(handle, &fp, filter_exp, 0, PCAP_NETMASK_UNKNOWN) == -1) {
    fprintf(stderr, "Couldn't parse filter %s: %s\n", filter_exp, pcap_geterr(handle));
    return 1;
}

// 应用过滤器
if (pcap_setfilter(handle, &fp) == -1) {
    fprintf(stderr, "Couldn't install filter %s: %s\n", filter_exp, pcap_geterr(handle));
    return 1;
}

// 捕获数据包
void packet_handler(u_char *user_data, const struct pcap_pkthdr *header, const u_char *packet) {
    printf("Packet length: %d\n", header->len);
    printf("Capture time: %s\n", ctime((const time_t *)&header->ts.tv_sec));

    // 解析以太网头部
    struct ether_header *eth = (struct ether_header *)packet;
    if (ntohs(eth->ether_type) == ETHERTYPE_IP) {
        // 解析 IP 头部
        struct iphdr *ip = (struct iphdr *)(packet + sizeof(struct ether_header));
        printf("Source IP: %s\n", inet_ntoa(*(struct in_addr *)&ip->saddr));
        printf("Dest IP: %s\n", inet_ntoa(*(struct in_addr *)&ip->daddr));
    }
}

// 主函数
int main() {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle;

    // 打开网络接口
    handle = pcap_open_live("eth0", BUFSIZ, 1, 1000, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "Couldn't open device: %s\n", errbuf);
        return 1;
    }

    // 开始捕获
    pcap_loop(handle, 10, packet_handler, NULL);

    // 清理
    pcap_freecode(&fp);
    pcap_close(handle);
    return 0;
}
```

### 23.9.3 BPF 过滤器原理

```
BPF (Berkeley Packet Filter) 架构:

  用户空间:
    过滤表达式: "tcp port 80"
        │
        ▼
    BPF 编译器
        │
        ▼
    BPF 指令序列
        │
        ▼
  ────────────────────────────
  内核空间:
        │
        ▼
    BPF 虚拟机
    ┌─────────────────────┐
    │  寄存器: A, X       │
    │  内存: M[0]-M[15]   │
    │  PC: 程序计数器      │
    └─────────────────────┘
        │
        ▼
    数据包匹配？
    ├── 是 → 接受数据包
    └── 否 → 丢弃数据包

BPF 指令示例 (tcp port 80):
  ld   [12]          ; 加载以太网类型字段
  jeq  #0x800        ; 是否是 IPv4?
  jf   drop          ; 不是则丢弃
  ld   [23]          ; 加载 IP 协议字段
  jeq  #6            ; 是否是 TCP?
  jf   drop          ; 不是则丢弃
  ldh  [20]          ; 加载 IP 标志和片偏移
  jset #0x1fff       ; 是否是分片?
  jf   drop          ; 是分片则丢弃
  ldxb 4*([14]&0xf)  ; 计算 IP 头部长度
  ldh  [x + 14]      ; 加载源端口
  jeq  #80           ; 源端口是 80?
  jt   accept        ; 是则接受
  ldh  [x + 16]      ; 加载目标端口
  jeq  #80           ; 目标端口是 80?
  jt   accept        ; 是则接受
  drop: ret #0       ; 丢弃
  accept: ret #65535 ; 接受
```

<div data-component="BPFExplorer"></div>

---

## 23.10 Python 网络编程

### 23.10.1 Python TCP 服务器

```python
import socket
import threading

class TCPServer:
    """Python TCP 服务器"""
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.server_socket = None

    def start(self):
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(128)

        print(f"Server listening on {self.host}:{self.port}")

        while True:
            client_socket, address = self.server_socket.accept()
            print(f"New connection from {address}")
            # 为每个客户端创建新线程
            thread = threading.Thread(target=self._handle_client,
                                     args=(client_socket, address))
            thread.daemon = True
            thread.start()

    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """处理客户端连接"""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                print(f"Received from {address}: {data.decode()}")
                # 回显数据
                client_socket.sendall(data)
        except ConnectionResetError:
            pass
        finally:
            client_socket.close()
            print(f"Connection closed: {address}")

if __name__ == '__main__':
    server = TCPServer()
    server.start()
```

### 23.10.2 Python TCP 客户端

```python
import socket

class TCPClient:
    """Python TCP 客户端"""
    def __init__(self, host: str = '127.0.0.1', port: int = 8080):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """连接服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def send(self, data: str) -> str:
        """发送数据并接收响应"""
        self.socket.sendall(data.encode())
        response = self.socket.recv(4096)
        return response.decode()

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()

if __name__ == '__main__':
    client = TCPClient()
    client.connect()

    while True:
        message = input("Enter message (or 'quit' to exit): ")
        if message.lower() == 'quit':
            break
        response = client.send(message)
        print(f"Server response: {response}")

    client.close()
```

### 23.10.3 Python asyncio 高性能服务器

```python
import asyncio

class AsyncTCPServer:
    """异步 TCP 服务器"""
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port

    async def handle_client(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter):
        """处理客户端连接"""
        addr = writer.get_extra_info('peername')
        print(f"New connection from {addr}")

        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break

                message = data.decode()
                print(f"Received from {addr}: {message}")

                # 回显数据
                writer.write(data)
                await writer.drain()
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Connection closed: {addr}")

    async def start(self):
        """启动服务器"""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )

        addr = server.sockets[0].getsockname()
        print(f"Async server listening on {addr}")

        async with server:
            await server.serve_forever()

if __name__ == '__main__':
    server = AsyncTCPServer()
    asyncio.run(server.start())
```

<div data-component="PythonSocketDemo"></div>

---

## 23.11 章节小结

本章详细介绍了套接字编程与网络应用开发的各个方面：

1. **套接字 API**：socket/bind/listen/accept/connect/read/write/close
2. **TCP 编程**：客户端/服务器流程、状态转换
3. **UDP 编程**：无连接通信、sendto/recvfrom
4. **并发服务器**：多进程、多线程、线程池
5. **I/O 多路复用**：select、poll、epoll 的原理与对比
6. **事件驱动**：Reactor/Proactor 模式
7. **连接管理**：连接池、超时处理
8. **原始套接字**：数据包捕获与 BPF 过滤器
9. **Python 网络编程**：同步与异步实现

<div data-component="ChapterSummary"></div>
<div data-component="KnowledgeCheck"></div>
