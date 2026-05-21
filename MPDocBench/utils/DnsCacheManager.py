import socket
import time
from typing import Dict, List, Optional, Tuple
from functools import wraps


class DnsCacheEntry:
    """DNS 缓存条目（支持 TTL 和永不过期）"""
    NEVER_EXPIRE = -1

    def __init__(self, ips: List[str], ttl: int = NEVER_EXPIRE):
        self.ips = ips
        self.ttl = ttl
        self.created_at = time.time() if ttl != self.NEVER_EXPIRE else 0

    def is_expired(self) -> bool:
        if self.ttl == self.NEVER_EXPIRE:
            return False
        return (time.time() - self.created_at) > self.ttl

    def __repr__(self):
        status = "NEVER_EXPIRE" if self.ttl == self.NEVER_EXPIRE else f"expires_in:{self.ttl}s"
        return f"DnsCacheEntry(ips={self.ips}, {status})"


class DnsCacheManager:
    """Python 版 DnsCacheManipulator - 应用级 DNS 缓存控制"""

    _cache: Dict[str, DnsCacheEntry] = {}
    _negative_cache: Dict[str, float] = {}  # host -> expiry_time
    _negative_ttl = 10  # 负缓存默认 10 秒
    _original_getaddrinfo = socket.getaddrinfo

    @classmethod
    def set_dns_cache(cls, host: str, *ips: str, ttl: int = DnsCacheEntry.NEVER_EXPIRE):
        """
        设置永不过期（或指定 TTL）的 DNS 缓存

        等效 Java: DnsCacheManipulator.setDnsCache(host, ips...)
        """
        if not ips:
            raise ValueError("至少需要一个 IP 地址")
        cls._cache[host] = DnsCacheEntry(list(ips), ttl)
        print(f"✓ set_dns_cache: {host} -> {ips} (ttl={ttl if ttl != DnsCacheEntry.NEVER_EXPIRE else 'NEVER'})")

    @classmethod
    def set_negative_cache(cls, host: str, ttl: int = None):
        """设置负缓存（模拟解析失败）"""
        cls._negative_cache[host] = time.time() + (ttl or cls._negative_ttl)
        print(f"✓ set_negative_cache: {host} (blocked for {ttl or cls._negative_ttl}s)")

    @classmethod
    def remove_dns_cache(cls, host: str):
        """清除指定 host 的缓存"""
        cls._cache.pop(host, None)
        cls._negative_cache.pop(host, None)
        print(f"✓ remove_dns_cache: {host}")

    @classmethod
    def clear_cache(cls):
        """清空所有缓存"""
        cls._cache.clear()
        cls._negative_cache.clear()
        print("✓ clear_cache: all entries removed")

    @classmethod
    def get_cache(cls, host: str) -> Optional[DnsCacheEntry]:
        """获取缓存条目（自动清理过期项）"""
        entry = cls._cache.get(host)
        if entry and entry.is_expired():
            cls._cache.pop(host, None)
            return None
        return entry

    @classmethod
    def _patched_getaddrinfo(cls, host, port, family=0, type=0, proto=0, flags=0):
        """拦截 DNS 查询，优先返回缓存结果"""

        # 1. 检查负缓存
        neg_expire = cls._negative_cache.get(host)
        if neg_expire and time.time() < neg_expire:
            raise socket.gaierror(11001, f"模拟 DNS 解析失败: {host} (负缓存中)")
        elif neg_expire and time.time() >= neg_expire:
            cls._negative_cache.pop(host, None)

        # 2. 检查正缓存
        entry = cls.get_cache(host)
        if entry:
            # 返回第一个 IP 的标准 getaddrinfo 结果
            fake_host = entry.ips[0]
            return cls._original_getaddrinfo(fake_host, port, family, type, proto, flags)

        # 3. 无缓存，走系统解析
        return cls._original_getaddrinfo(host, port, family, type, proto, flags)

    @classmethod
    def install_patch(cls):
        """激活 DNS 缓存拦截（必须在所有网络操作前调用）"""
        if socket.getaddrinfo == cls._patched_getaddrinfo:
            print("⚠ DNS cache patch 已激活，跳过重复安装")
            return

        socket.getaddrinfo = cls._patched_getaddrinfo
        print("✓ DNS cache patch 已激活")

    @classmethod
    def uninstall_patch(cls):
        """恢复原始 DNS 行为"""
        socket.getaddrinfo = cls._original_getaddrinfo
        print("✓ DNS cache patch 已卸载")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 1. 激活 patch（必须最先执行！）
    DnsCacheManager.install_patch()

    # 2. 设置永不过期的 DNS 映射
    DnsCacheManager.set_dns_cache("api.example.com", "127.0.0.1", "127.0.0.2")

    # 3. 设置带 TTL 的缓存（30 秒后过期）
    DnsCacheManager.set_dns_cache("temp.service", "192.168.1.100", ttl=30)

    # 4. 测试解析
    print("\n>>> 测试解析 api.example.com")
    addr_info = socket.getaddrinfo("api.example.com", 80)
    print(f"  结果: {addr_info[0][4][0]}")  # 应输出 127.0.0.1

    # 5. 验证 requests 库也受影响
    import requests

    try:
        resp = requests.get("http://api.example.com:8000/health", timeout=2)
        print(f"  requests 访问成功: {resp.status_code}")
    except Exception as e:
        print(f"  requests 访问异常（预期，因本地无服务）: {type(e).__name__}")

    # 6. 查看当前缓存
    print("\n>>> 当前缓存状态")
    for host, entry in DnsCacheManager._cache.items():
        print(f"  {host}: {entry}")

    # 7. 清理
    # DnsCacheManager.remove_dns_cache("api.example.com")
    # DnsCacheManager.uninstall_patch()