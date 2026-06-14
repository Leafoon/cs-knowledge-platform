# Chapter 22: 应用层 — P2P 与 CDN 技术

> **学习目标**：
> - 理解 P2P 与客户端-服务器架构的区别与优劣
> - 掌握 BitTorrent 协议的核心机制：元文件、Tracker、Peer Wire、分片选择
> - 理解 DHT（分布式哈希表）的工作原理：Chord、Kademlia
> - 掌握 CDN 的架构：DNS 重定向、边缘服务器、内容复制
> - 理解自适应比特率流媒体（ABR）的工作原理
> - 掌握视频流媒体的缓冲区管理策略
> - 了解直播流媒体的技术挑战与解决方案

---

## 22.1 P2P 与客户端-服务器对比

### 22.1.1 架构对比

```
客户端-服务器 (C/S) 架构:

  ┌────────┐  ┌────────┐  ┌────────┐
  │Client 1│  │Client 2│  │Client 3│
  └───┬────┘  └───┬────┘  └───┬────┘
      │           │           │
      └───────────┼───────────┘
                  │
            ┌─────┴─────┐
            │  服务器    │  ← 单点瓶颈
            └───────────┘

P2P 架构:

  ┌────────┐◄──►┌────────┐
  │Peer 1  │    │Peer 2  │
  └───┬────┘    └────┬───┘
      │     ▲  ▲     │
      │     │  │     │
      ▼     │  │     ▼
  ┌────────┐│  │┌────────┐
  │Peer 3  │◄┘  └►│Peer 4  │
  └────────┘      └────────┘

  每个节点既是客户端又是服务器
  没有单点瓶颈
```

### 22.1.2 性能对比

```
文件分发场景：N 个客户端下载一个 F 字节的文件

C/S 架构:
  服务器上传时间: N * F / us (us: 服务器上传带宽)
  最慢客户端下载时间: F / dmin (dmin: 最慢客户端带宽)
  总时间: max(N*F/us, F/dmin)
  → 服务器成为瓶颈，N 越大越慢

P2P 架构:
  初始上传: F / us (服务器上传一次)
  对等体之间: 每个对等体下载后可以上传
  总时间: max(F/us, F/dmin, N*F/(us + Σdi))
  → 系统总带宽随 N 增加而增加

P2P 的"越多人越快"特性:
  N=1:  P2P ≈ C/S
  N=100: P2P 比 C/S 快 10-50 倍
  N=10000: P2P 比 C/S 快 1000 倍
```

<div data-component="P2PvsCSChart"></div>

---

## 22.2 BitTorrent 协议

### 22.2.1 核心概念

```
BitTorrent 术语:
  - Torrent: 一个文件分发任务
  - .torrent 文件: 元数据文件（包含 Tracker URL、文件哈希等）
  - Tracker: 跟踪参与下载的对等体
  - Peer: 参与下载的对等体
  - Seed: 拥有完整文件的对等体
  - Leecher: 正在下载的对等体
  - Piece: 文件被分成的大块（通常 256KB-4MB）
  - Block: Piece 被分成的小块（通常 16KB）
  - Swarm: 下载同一个文件的所有对等体
```

### 22.2.2 Torrent 文件格式

```python
import bencodepy

def parse_torrent_file(filepath: str) -> dict:
    """解析 .torrent 文件"""
    with open(filepath, 'rb') as f:
        data = bencodepy.decode(f.read())

    torrent = {
        'announce': data[b'announce'].decode(),      # Tracker URL
        'created_by': data.get(b'created by', b'').decode(),
        'creation_date': data.get(b'creation date', 0),
        'info': {},  # 文件信息
    }

    info = data[b'info']
    torrent['info']['name'] = info[b'name'].decode()
    torrent['info']['piece_length'] = info[b'piece length']
    torrent['info']['pieces'] = info[b'pieces']  # 所有 piece 的 SHA1 哈希

    if b'files' in info:
        # 多文件模式
        torrent['info']['files'] = []
        for f in info[b'files']:
            torrent['info']['files'].append({
                'length': f[b'length'],
                'path': '/'.join(p.decode() for p in f[b'path'])
            })
        torrent['info']['total_length'] = sum(f['length'] for f in torrent['info']['files'])
    else:
        # 单文件模式
        torrent['info']['length'] = info[b'length']
        torrent['info']['total_length'] = info[b'length']

    # 计算 info_hash
    import hashlib
    info_encoded = bencodepy.encode(info)
    torrent['info_hash'] = hashlib.sha1(info_encoded).digest()

    return torrent
```

### 22.2.3 Tracker 协议

```python
import requests
import struct

class BitTorrentTracker:
    """BitTorrent Tracker 客户端"""
    def __init__(self, torrent: dict, peer_id: bytes, port: int = 6881):
        self.torrent = torrent
        self.peer_id = peer_id
        self.port = port

    def announce(self, uploaded: int, downloaded: int, left: int, event: str = 'started') -> list:
        """向 Tracker 发送 announce 请求"""
        params = {
            'info_hash': self.torrent['info_hash'],
            'peer_id': self.peer_id,
            'port': self.port,
            'uploaded': uploaded,
            'downloaded': downloaded,
            'left': left,
            'event': event,
            'compact': 1,  # 紧凑格式
            'numwant': 200,  # 期望的对等体数量
        }

        response = requests.get(self.torrent['announce'], params=params, timeout=10)
        data = bencodepy.decode(response.content)

        peers = self._parse_peers(data[b'peers'])
        return peers

    def _parse_peers(self, peers_data) -> list:
        """解析紧凑格式的对等体列表"""
        peers = []
        # 紧凑格式：每个对等体 6 字节（4 字节 IP + 2 字节端口）
        for i in range(0, len(peers_data), 6):
            ip = '.'.join(str(b) for b in peers_data[i:i+4])
            port = struct.unpack('!H', peers_data[i+4:i+6])[0]
            peers.append((ip, port))
        return peers
```

### 22.2.4 Peer Wire 协议

```
Peer Wire 协议消息类型:

握手 (Handshake):
  <pstrlen><pstr><reserved><info_hash><peer_id>
  pstrlen = 19, pstr = "BitTorrent protocol"

消息格式:
  <length><id><payload>

消息类型:
  0 - choke: 阻塞对方
  1 - unchoke: 解除阻塞
  2 - interested: 表示感兴趣
  3 - not interested: 表示不感兴趣
  4 - have: 通告拥有某个 piece
  5 - bitfield: 通告拥有的所有 pieces
  6 - request: 请求一个 block
  7 - piece: 发送一个 block 的数据
  8 - cancel: 取消请求
```

### 22.2.5 分片选择引擎

```python
class PieceSelector:
    """BitTorrent 分片选择引擎"""
    def __init__(self, total_pieces: int, piece_length: int):
        self.total_pieces = total_pieces
        self.piece_length = piece_length
        self.have = [False] * total_pieces        # 本地拥有的 pieces
        self.needed = [True] * total_pieces        # 还需要的 pieces
        self.peer_bitfields = {}                   # {peer_id: [bool] * total_pieces}
        self.piece_availability = [0] * total_pieces  # 每个 piece 的可用性

    def update_peer_bitfield(self, peer_id: str, bitfield: list):
        """更新对等体的 bitfield"""
        old = self.peer_bitfields.get(peer_id, [False] * self.total_pieces)
        self.peer_bitfields[peer_id] = bitfield

        # 更新可用性计数
        for i in range(self.total_pieces):
            if bitfield[i] and not old[i]:
                self.piece_availability[i] += 1
            elif not bitfield[i] and old[i]:
                self.piece_availability[i] -= 1

    def select_pieces_rarest_first(self, count: int = 4) -> list:
        """最稀缺优先策略"""
        candidates = []
        for i in range(self.total_pieces):
            if self.needed[i] and not self.have[i] and self.piece_availability[i] > 0:
                candidates.append((self.piece_availability[i], i))

        # 按可用性排序（最稀缺的优先）
        candidates.sort(key=lambda x: x[0])
        return [idx for _, idx in candidates[:count]]

    def select_pieces_random_first(self, count: int = 4) -> list:
        """随机优先策略（初始阶段）"""
        import random
        candidates = [i for i in range(self.total_pieces)
                      if self.needed[i] and not self.have[i] and self.piece_availability[i] > 0]
        random.shuffle(candidates)
        return candidates[:count]

    def select_endgame_piece(self) -> Optional[int]:
        """终局模式：请求所有剩余的 pieces"""
        for i in range(self.total_pieces):
            if self.needed[i] and not self.have[i]:
                return i
        return None

    def mark_piece_complete(self, piece_idx: int):
        """标记 piece 完成"""
        self.have[piece_idx] = True
        self.needed[piece_idx] = False
        # 验证哈希
        if not self._verify_piece(piece_idx):
            self.have[piece_idx] = False
            self.needed[piece_idx] = True
            raise ValueError(f"Piece {piece_idx} hash verification failed")

    def _verify_piece(self, piece_idx: int) -> bool:
        """验证 piece 的 SHA1 哈希"""
        import hashlib
        piece_data = self._read_piece(piece_idx)
        expected_hash = self._get_expected_hash(piece_idx)
        actual_hash = hashlib.sha1(piece_data).digest()
        return actual_hash == expected_hash


class ChokingAlgorithm:
    """BitTorrent 阻塞算法"""
    def __init__(self, max_uploads: int = 4):
        self.max_uploads = max_uploads
        self.peers = {}           # {peer_id: PeerInfo}
        self.optimistic_unchoke = None  # 乐观解阻塞的对等体
        self.unchoked = set()     # 当前解阻塞的对等体

    def update_peer(self, peer_id: str, download_rate: float, upload_rate: float, is_interested: bool):
        """更新对等体状态"""
        self.peers[peer_id] = {
            'download_rate': download_rate,
            'upload_rate': upload_rate,
            'is_interested': is_interested,
            'is_choking': peer_id not in self.unchoked,
        }

    def recalculate_unchoked(self):
        """重新计算解阻塞的对等体"""
        # 按下载速率排序（我们从他们下载最快的优先）
        interested = [(pid, info) for pid, info in self.peers.items()
                      if info['is_interested']]
        interested.sort(key=lambda x: x[1]['download_rate'], reverse=True)

        # 选择前 max_uploads 个
        self.unchoked = set(pid for pid, _ in interested[:self.max_uploads])

        # 乐观解阻塞（每 3 轮换一次）
        if self.optimistic_unchoke is None or random.random() < 0.33:
            choked = [pid for pid, info in self.peers.items()
                      if info['is_interested'] and pid not in self.unchoked]
            if choked:
                self.optimistic_unchoke = random.choice(choked)
                self.unchoked.add(self.optimistic_unchoke)

        return self.unchoked
```

<div data-component="BitTorrentDemo"></div>

---

## 22.3 BitTorrent 客户端的对等节点管理器

### 22.3.1 对等节点管理

```python
from dataclasses import dataclass
from typing import Dict, List, Set
import time

@dataclass
class PeerInfo:
    peer_id: str
    ip: str
    port: int
    bitfield: List[bool]
    am_choking: bool = True        # 我们是否阻塞对方
    am_interested: bool = False    # 我们是否对对方感兴趣
    peer_choking: bool = True      # 对方是否阻塞我们
    peer_interested: bool = False  # 对方是否对我们感兴趣
    download_rate: float = 0.0     # 下载速率 (bytes/s)
    upload_rate: float = 0.0       # 上传速率 (bytes/s)
    last_seen: float = 0.0         # 最后活跃时间
    connected_at: float = 0.0      # 连接建立时间
    pending_requests: int = 0      # 待处理的请求数

class PeerManager:
    """对等节点管理器"""
    def __init__(self, max_peers: int = 50):
        self.max_peers = max_peers
        self.peers: Dict[str, PeerInfo] = {}
        self.connecting: Set[str] = set()
        self.banned: Set[str] = set()

    def add_peer(self, ip: str, port: int) -> Optional[str]:
        """添加新的对等体"""
        peer_id = f"{ip}:{port}"

        if peer_id in self.banned:
            return None

        if len(self.peers) >= self.max_peers:
            # 淘汰最慢的对等体
            self._evict_slowest()

        if peer_id not in self.peers:
            self.peers[peer_id] = PeerInfo(
                peer_id=peer_id,
                ip=ip,
                port=port,
                bitfield=[],
                last_seen=time.time(),
                connected_at=time.time(),
            )
            self.connecting.add(peer_id)

        return peer_id

    def remove_peer(self, peer_id: str):
        """移除对等体"""
        if peer_id in self.peers:
            del self.peers[peer_id]
        self.connecting.discard(peer_id)

    def ban_peer(self, peer_id: str, reason: str):
        """封禁对等体"""
        self.banned.add(peer_id)
        self.remove_peer(peer_id)

    def get_best_peers(self, count: int = 10) -> List[PeerInfo]:
        """获取最佳的对等体（用于下载）"""
        candidates = [p for p in self.peers.values()
                      if not p.peer_choking and p.am_interested]
        candidates.sort(key=lambda p: p.download_rate, reverse=True)
        return candidates[:count]

    def get_upload_peers(self, count: int = 10) -> List[PeerInfo]:
        """获取用于上传的对等体"""
        candidates = [p for p in self.peers.values()
                      if p.peer_interested]
        candidates.sort(key=lambda p: p.upload_rate, reverse=True)
        return candidates[:count]

    def _evict_slowest(self):
        """淘汰最慢的对等体"""
        if not self.peers:
            return
        slowest = min(self.peers.values(), key=lambda p: p.download_rate)
        self.remove_peer(slowest.peer_id)

    def update_rates(self, peer_id: str, downloaded: int, uploaded: int, interval: float):
        """更新对等体的传输速率"""
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            peer.download_rate = downloaded / interval
            peer.upload_rate = uploaded / interval
            peer.last_seen = time.time()
```

<div data-component="PeerManagerDemo"></div>

---

## 22.4 DHT 分布式哈希表

### 22.4.1 DHT 概述

DHT（Distributed Hash Table）是一种去中心化的键值存储系统，用于在没有 Tracker 的情况下发现对等体：

```
DHT 的作用:
  传统 BitTorrent: 需要 Tracker 服务器协调对等体
  DHT BitTorrent:  对等体自组织，无需中心服务器

  DHT 存储: key = info_hash, value = 对等体列表
  查询: 给定 info_hash，找到拥有该文件的对等体
```

### 22.4.2 Chord 协议

```
Chord 环形拓扑:

  假设节点 ID 为 m 位（通常 m=160，使用 SHA1）

  节点 ID: SHA1(IP:Port)
  键 ID: SHA1(key)

  每个节点维护:
    - 后继节点（successor）
    - 前驱节点（predecessor）
    - 指针表（finger table）

  指针表:
    finger[i] = 第一个 ID ≥ (n + 2^i) mod 2^m 的节点

  查找过程:
    要查找键 k，将请求转发给指针表中最大的不超过 k 的节点
    O(log N) 跳即可找到目标
```

```python
import hashlib

class ChordNode:
    """Chord DHT 节点"""
    def __init__(self, ip: str, port: int, m: int = 160):
        self.ip = ip
        self.port = port
        self.m = m
        self.node_id = self._hash(f"{ip}:{port}")
        self.successor = None
        self.predecessor = None
        self.finger_table = [None] * m
        self.data = {}  # 本地存储的键值对

    def _hash(self, key: str) -> int:
        """计算 SHA1 哈希"""
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def find_successor(self, id: int) -> 'ChordNode':
        """查找 ID 的后继节点"""
        if self._in_range(id, self.node_id, self.successor.node_id):
            return self.successor
        else:
            next_node = self._closest_preceding_node(id)
            return next_node.find_successor(id)

    def _closest_preceding_node(self, id: int) -> 'ChordNode':
        """查找最接近 ID 的前驱节点"""
        for i in range(self.m - 1, -1, -1):
            if self.finger_table[i] and self._in_range(self.finger_table[i].node_id, self.node_id, id):
                return self.finger_table[i]
        return self

    def _in_range(self, id: int, start: int, end: int) -> bool:
        """检查 ID 是否在 [start, end) 范围内"""
        if start < end:
            return start <= id < end
        else:
            return start <= id or id < end

    def put(self, key: str, value: str):
        """存储键值对"""
        key_hash = self._hash(key)
        responsible = self.find_successor(key_hash)
        responsible.data[key] = value

    def get(self, key: str) -> Optional[str]:
        """查询键值对"""
        key_hash = self._hash(key)
        responsible = self.find_successor(key_hash)
        return responsible.data.get(key)

    def join(self, existing_node: 'ChordNode'):
        """加入 DHT 网络"""
        self.predecessor = None
        self.successor = existing_node.find_successor(self.node_id)
        self._update_finger_table()

    def stabilize(self):
        """定期稳定化"""
        x = self.successor.predecessor
        if x and self._in_range(x.node_id, self.node_id, self.successor.node_id):
            self.successor = x
        self.successor.notify(self)

    def notify(self, node: 'ChordNode'):
        """被通知可能是前驱"""
        if self.predecessor is None or self._in_range(node.node_id, self.predecessor.node_id, self.node_id):
            self.predecessor = node
```

### 22.4.3 Kademlia 协议

```python
import hashlib
import random
from typing import List, Tuple

class KademliaNode:
    """Kademlia DHT 节点"""
    def __init__(self, ip: str, port: int, k: int = 20):
        self.ip = ip
        self.port = port
        self.k = k  # 每个桶的最大节点数
        self.node_id = self._generate_id()
        self.routing_table = [[] for _ in range(160)]  # K 桶
        self.data = {}  # 存储的键值对

    def _generate_id(self) -> int:
        """生成 160 位随机 ID"""
        return random.getrandbits(160)

    def _distance(self, id1: int, id2: int) -> int:
        """计算两个 ID 的 XOR 距离"""
        return id1 ^ id2

    def _bucket_index(self, id: int) -> int:
        """计算目标 ID 应该放入哪个桶"""
        distance = self._distance(self.node_id, id)
        if distance == 0:
            return 0
        return distance.bit_length() - 1

    def update_routing_table(self, node_id: int, ip: str, port: int):
        """更新路由表"""
        bucket_idx = self._bucket_index(node_id)
        bucket = self.routing_table[bucket_idx]

        # 检查节点是否已在桶中
        for i, (nid, nip, nport) in enumerate(bucket):
            if nid == node_id:
                # 移到末尾（最近使用）
                bucket.pop(i)
                bucket.append((node_id, ip, port))
                return

        if len(bucket) < self.k:
            # 桶未满，直接添加
            bucket.append((node_id, ip, port))
        else:
            # 桶已满，ping 最老的节点
            oldest = bucket[0]
            if not self._ping(oldest[1], oldest[2]):
                # 最老的节点不可达，替换
                bucket.pop(0)
                bucket.append((node_id, ip, port))

    def find_node(self, target_id: int) -> List[Tuple[int, str, int]]:
        """查找最接近 target_id 的 k 个节点"""
        bucket_idx = self._bucket_index(target_id)
        candidates = []

        # 从目标桶开始，向两边扩展
        for i in range(160):
            if bucket_idx + i < 160:
                candidates.extend(self.routing_table[bucket_idx + i])
            if bucket_idx - i >= 0:
                candidates.extend(self.routing_table[bucket_idx - i])
            if len(candidates) >= self.k:
                break

        # 按距离排序
        candidates.sort(key=lambda x: self._distance(x[0], target_id))
        return candidates[:self.k]

    def store(self, key: str, value: str):
        """存储键值对"""
        key_hash = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        closest = self.find_node(key_hash)
        # 将值存储到最接近的 k 个节点
        for node_id, ip, port in closest:
            self._send_store(ip, port, key_hash, value)

    def lookup(self, key: str) -> Optional[str]:
        """查询键值对"""
        key_hash = int(hashlib.sha1(key.encode()).hexdigest(), 16)

        # 迭代查找
        closest = self.find_node(key_hash)
        visited = set()

        for _ in range(20):  # 最多迭代 20 次
            new_closest = []
            for node_id, ip, port in closest:
                if node_id in visited:
                    continue
                visited.add(node_id)

                # 查询该节点
                result = self._send_find_value(ip, port, key_hash)
                if result and isinstance(result, str):
                    return result  # 找到值
                elif result:
                    new_closest.extend(result)

            if not new_closest:
                break

            # 更新最近节点列表
            new_closest.sort(key=lambda x: self._distance(x[0], key_hash))
            closest = new_closest[:self.k]

        return None
```

<div data-component="DHTDemo"></div>

---

## 22.5 CDN 内容分发网络

### 22.5.1 CDN 架构

```
CDN 全局架构:

  用户 (北京)  用户 (上海)  用户 (广州)
      │            │            │
      ▼            ▼            ▼
  ┌───────┐   ┌───────┐   ┌───────┐
  │边缘节点│   │边缘节点│   │边缘节点│  ← 就近服务
  │(北京) │   │(上海) │   │(广州) │
  └───┬───┘   └───┬───┘   └───┬───┘
      │           │           │
      └───────────┼───────────┘
                  │
           ┌──────┴──────┐
           │   源站       │  ← 内容源
           │(Origin)     │
           └─────────────┘

工作流程:
  1. 用户请求 www.example.com/image.jpg
  2. DNS 将用户重定向到最近的边缘节点
  3. 边缘节点检查缓存
  4. 如果缓存命中，直接返回
  5. 如果缓存未命中，从源站获取并缓存
```

### 22.5.2 CDN 全局负载均衡器的 DNS 重定向策略

```python
import geoip2.database
from typing import List, Tuple

class CDNLoadBalancer:
    """CDN 全局负载均衡器"""
    def __init__(self, edge_servers: List[dict]):
        self.edge_servers = edge_servers  # [{ip, region, capacity, load}]
        self.geo_reader = geoip2.database.Reader('GeoLite2-City.mmdb')

    def resolve(self, client_ip: str, domain: str) -> str:
        """DNS 重定向：将用户导向最近的边缘节点"""
        # 1. 地理位置查询
        client_location = self._get_location(client_ip)

        # 2. 计算每个边缘节点的分数
        scores = []
        for server in self.edge_servers:
            score = self._calculate_score(server, client_location)
            scores.append((score, server))

        # 3. 选择最佳节点
        scores.sort(key=lambda x: x[0], reverse=True)
        best_server = scores[0][1]

        return best_server['ip']

    def _get_location(self, ip: str) -> dict:
        """获取 IP 的地理位置"""
        try:
            response = self.geo_reader.city(ip)
            return {
                'latitude': response.location.latitude,
                'longitude': response.location.longitude,
                'country': response.country.iso_code,
                'city': response.city.name,
            }
        except:
            return {'latitude': 0, 'longitude': 0, 'country': 'US', 'city': 'Unknown'}

    def _calculate_score(self, server: dict, client_location: dict) -> float:
        """计算边缘节点的分数"""
        # 1. 距离分数（距离越近越好）
        distance = self._haversine(
            client_location['latitude'], client_location['longitude'],
            server['latitude'], server['longitude']
        )
        distance_score = 1.0 / (1.0 + distance / 1000)  # 归一化

        # 2. 负载分数（负载越低越好）
        load_ratio = server['load'] / server['capacity']
        load_score = 1.0 - load_ratio

        # 3. 健康分数
        health_score = 1.0 if server['healthy'] else 0.0

        # 综合分数（加权平均）
        score = (0.5 * distance_score +
                 0.3 * load_score +
                 0.2 * health_score)

        return score

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点之间的距离（公里）"""
        import math
        R = 6371  # 地球半径

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c
```

<div data-component="CDNArchitectureDemo"></div>

### 22.5.3 内容复制策略

```
CDN 内容复制策略:

1. 拉模式 (Pull)
   - 内容按需获取
   - 用户首次请求时从源站拉取
   - 优点：不需要预复制
   - 缺点：首次请求慢

2. 推模式 (Push)
   - 内容主动推送到边缘节点
   - 适合热门内容
   - 优点：首次请求快
   - 缺点：需要预测需求

3. 混合模式
   - 热门内容推模式
   - 冷门内容拉模式
   - 根据访问频率动态调整
```

---

## 22.6 视频流媒体技术

### 22.6.1 DASH（Dynamic Adaptive Streaming over HTTP）

```
DASH 工作原理:
  1. 视频被编码为多个比特率版本
  2. 每个版本被分割成小片段（2-10 秒）
  3. 客户端根据网络状况选择比特率
  4. 每个片段通过 HTTP 独立请求

DASH 清单文件（MPD）:
  <MPD>
    <Period>
      <AdaptationSet mimeType="video/mp4">
        <Representation bandwidth="500000"  height="360">
          <BaseURL>video_360p/</BaseURL>
          <SegmentTemplate media="seg$Number$.m4s" startNumber="1"/>
        </Representation>
        <Representation bandwidth="2000000" height="720">
          <BaseURL>video_720p/</BaseURL>
          <SegmentTemplate media="seg$Number$.m4s" startNumber="1"/>
        </Representation>
        <Representation bandwidth="5000000" height="1080">
          <BaseURL>video_1080p/</BaseURL>
          <SegmentTemplate media="seg$Number$.m4s" startNumber="1"/>
        </Representation>
      </AdaptationSet>
    </Period>
  </MPD>

可用比特率:
  360p:  500 Kbps  (低质量，适合慢速网络)
  720p:  2 Mbps    (高清)
  1080p: 5 Mbps    (全高清)
  4K:    20 Mbps   (超高清)
```

### 22.6.2 自适应比特率（ABR）算法

```python
class ABRController:
    """自适应比特率控制器"""
    def __init__(self, bitrate_levels: list):
        self.bitrate_levels = sorted(bitrate_levels)  # [500000, 2000000, 5000000]
        self.current_bitrate = bitrate_levels[0]
        self.bandwidth_estimate = 0
        self.buffer_level = 0
        self.rebuffer_count = 0

    def select_bitrate(self, buffer_level: float, bandwidth_estimate: float) -> int:
        """选择下一个片段的比特率"""
        self.buffer_level = buffer_level
        self.bandwidth_estimate = bandwidth_estimate

        # 策略 1: 带宽估计法
        safe_bandwidth = bandwidth_estimate * 0.8  # 留 20% 余量
        bandwidth_selected = self._select_by_bandwidth(safe_bandwidth)

        # 策略 2: 缓冲区水位法
        buffer_selected = self._select_by_buffer(buffer_level)

        # 取两者中较低的（更保守）
        return min(bandwidth_selected, buffer_selected)

    def _select_by_bandwidth(self, bandwidth: float) -> int:
        """基于带宽估计选择比特率"""
        selected = self.bitrate_levels[0]
        for bitrate in self.bitrate_levels:
            if bitrate <= bandwidth:
                selected = bitrate
            else:
                break
        return selected

    def _select_by_buffer(self, buffer: float) -> int:
        """基于缓冲区水位选择比特率"""
        # 缓冲区阈值（秒）
        LOW_THRESHOLD = 5
        HIGH_THRESHOLD = 15

        if buffer < LOW_THRESHOLD:
            # 缓冲区低，使用最低比特率
            return self.bitrate_levels[0]
        elif buffer < HIGH_THRESHOLD:
            # 缓冲区中等，使用中等比特率
            return self.bitrate_levels[len(self.bitrate_levels) // 2]
        else:
            # 缓冲区充足，使用最高比特率
            return self.bitrate_levels[-1]

    def on_segment_downloaded(self, segment_size: int, download_time: float):
        """片段下载完成回调"""
        # 更新带宽估计（指数移动平均）
        measured_bandwidth = segment_size * 8 / download_time  # bps
        alpha = 0.3
        self.bandwidth_estimate = (alpha * measured_bandwidth +
                                   (1 - alpha) * self.bandwidth_estimate)


class BBAController:
    """Buffer-Based ABR (BBA) 算法"""
    def __init__(self, bitrate_levels: list, buffer_size: float = 30.0):
        self.bitrate_levels = sorted(bitrate_levels)
        self.buffer_size = buffer_size
        self.reservoir = 5.0      # 缓冲区下限（低于此使用最低码率）
        self.cushion = 20.0       # 缓冲区上限（高于此使用最高码率）

    def select_bitrate(self, buffer_level: float) -> int:
        """基于缓冲区选择比特率"""
        if buffer_level <= self.reservoir:
            return self.bitrate_levels[0]

        if buffer_level >= self.cushion:
            return self.bitrate_levels[-1]

        # 线性映射
        ratio = (buffer_level - self.reservoir) / (self.cushion - self.reservoir)
        index = int(ratio * (len(self.bitrate_levels) - 1))
        return self.bitrate_levels[min(index, len(self.bitrate_levels) - 1)]
```

<div data-component="ABRDemo"></div>

### 22.6.3 自适应比特率（ABR）流媒体的缓冲区管理器

```python
import time
from collections import deque
from typing import Optional

class StreamingBuffer:
    """流媒体缓冲区管理器"""
    def __init__(self, max_duration: float = 30.0, min_duration: float = 5.0):
        self.max_duration = max_duration      # 最大缓冲时长（秒）
        self.min_duration = min_duration      # 最小缓冲时长（秒）
        self.buffer = deque()                  # 缓冲区 [(data, duration, bitrate)]
        self.total_duration = 0.0              # 当前缓冲区总时长
        self.playback_position = 0.0           # 播放位置
        self.is_playing = False
        self.rebuffer_start = 0.0

    def add_segment(self, data: bytes, duration: float, bitrate: int):
        """添加片段到缓冲区"""
        if self.total_duration + duration > self.max_duration:
            raise BufferOverflowError("Buffer is full")

        self.buffer.append((data, duration, bitrate))
        self.total_duration += duration

        # 如果之前在重缓冲，现在有足够数据可以恢复
        if not self.is_playing and self.total_duration >= self.min_duration:
            self.is_playing = True
            rebuffer_time = time.time() - self.rebuffer_start
            return {'event': 'rebuffer_end', 'duration': rebuffer_time}

        return None

    def consume_segment(self) -> Optional[tuple]:
        """消费一个片段（播放）"""
        if not self.buffer:
            # 缓冲区为空，进入重缓冲
            self.is_playing = False
            self.rebuffer_start = time.time()
            return None

        data, duration, bitrate = self.buffer.popleft()
        self.total_duration -= duration
        self.playback_position += duration
        return (data, duration, bitrate)

    def get_buffer_level(self) -> float:
        """获取当前缓冲区水位（秒）"""
        return self.total_duration

    def get_health_status(self) -> dict:
        """获取缓冲区健康状态"""
        return {
            'buffer_level': self.total_duration,
            'is_playing': self.is_playing,
            'segments_queued': len(self.buffer),
            'is_critical': self.total_duration < self.min_duration,
            'is_full': self.total_duration >= self.max_duration * 0.9,
        }

    def predict_stall(self, bandwidth_estimate: float, next_segment_size: int) -> float:
        """预测是否会卡顿"""
        if not self.buffer:
            return 0.0  # 已经卡顿

        download_time = next_segment_size * 8 / bandwidth_estimate
        buffer_after_download = self.total_duration - download_time

        if buffer_after_download < 0:
            return abs(buffer_after_download)  # 预计卡顿时长
        return 0.0
```

<div data-component="BufferManagerDemo"></div>

---

## 22.7 直播流媒体

### 22.7.1 直播流媒体架构

```
直播流媒体架构:

  ┌─────────────┐
  │   主播端     │
  │ (推流)      │
  └──────┬──────┘
         │ RTMP/SRT
         ▼
  ┌─────────────┐
  │  流媒体服务器 │
  │ (转码/分发)  │
  └──────┬──────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  ┌───┐┌───┐┌───┐
  │CDN││CDN││CDN│
  └─┬─┘└─┬─┘└─┬─┘
    ▼    ▼    ▼
  观众1 观众2 观众3
```

### 22.7.2 直播协议对比

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│    协议       │    延迟      │    兼容性     │    用途       │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ RTMP         │ 1-3 秒       │ Flash/专用   │ 推流         │
│ HLS          │ 5-30 秒      │ 所有浏览器   │ 拉流         │
│ DASH         │ 3-10 秒      │ 所有浏览器   │ 拉流         │
│ WebRTC       │ < 1 秒       │ 现代浏览器   │ 实时互动     │
│ SRT          │ < 1 秒       │ 专用         │ 推流         │
│ LL-HLS       │ 2-5 秒       │ 现代浏览器   │ 低延迟拉流   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

<div data-component="LiveStreamingDemo"></div>

---

## 22.8 结构化 vs 非结构化覆盖网络

```
非结构化覆盖网络:
  - Gnutella, 早期 KaZaA
  - 使用洪泛查询
  - 查询 TTL 有限
  - 不保证找到内容
  - O(N) 消息复杂度

结构化覆盖网络:
  - Chord, Kademlia, Pastry
  - 使用分布式哈希表
  - O(log N) 查找
  - 保证找到存在的内容
  - 需要维护路由表
```

---

## 22.9 章节小结

本章详细介绍了 P2P 和 CDN 技术：

1. **P2P vs C/S**：P2P 的可扩展性优势与"越多人越快"特性
2. **BitTorrent**：元文件、Tracker、Peer Wire 协议、分片选择算法
3. **DHT**：Chord 环形拓扑和 Kademlia K 桶结构
4. **CDN**：DNS 重定向、边缘服务器、内容复制策略
5. **DASH 流媒体**：自适应比特率算法（带宽估计法、缓冲区水位法）
6. **缓冲区管理**：重缓冲预测、健康状态监控
7. **直播技术**：低延迟协议（WebRTC、LL-HLS）

<div data-component="ChapterSummary"></div>
<div data-component="KnowledgeCheck"></div>
