"""Read single members of a remote (optionally split, optionally nested) zip archive via HTTP range requests.

Only the central directory and the requested members are downloaded, so a few thousand images can be
taken out of a 40-250 GB archive on a small disk.
  - Split archives are given as their parts in disk order (`x.z01, x.z02, ..., x.zip`); member offsets are
    disk-relative, as in PKWARE split archives.
  - A zip stored (uncompressed) inside another zip is opened with `RemoteZip.nested(outer, name)`.
"""
import http.client
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib


def present(path):
    """Already fetched: exists and is non-empty. An interrupted write can leave a renamed but 0-byte file."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def _headers(extra=None):
    h = dict(extra or {})
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


class HttpFile:
    """A remote file read by byte ranges. The redirect target (signed CDN url) is cached and refreshed on 4xx."""

    def __init__(self, url):
        self.url, self._lock = url, threading.Lock()
        self._resolve()

    def _resolve(self):
        req = urllib.request.Request(self.url, headers=_headers({"Range": "bytes=0-0"}))
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            self.direct = r.geturl()

    def read(self, start, length):
        if length <= 0:
            return b""
        rng = {"Range": f"bytes={start}-{start + length - 1}"}
        for attempt in range(6):
            try:
                hdr = rng if self.direct != self.url else _headers(rng)  # never send the token to the CDN
                with urllib.request.urlopen(urllib.request.Request(self.direct, headers=hdr), timeout=120) as r:
                    data = r.read()
                if len(data) == length:
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404, 410):  # signed url expired
                    with self._lock:
                        self._resolve()
                elif attempt == 5:
                    raise
            except (OSError, http.client.HTTPException):  # IncompleteRead is an HTTPException, not an OSError
                if attempt == 5:
                    raise
            time.sleep(min(30, 2 ** attempt))
        raise IOError(f"short read {self.url} @{start}+{length}")


class Concat:
    """Several byte sources read as one (the parts of a split archive)."""

    def __init__(self, parts):
        self.parts = parts
        self.starts = [sum(p.size for p in parts[:i]) for i in range(len(parts))]
        self.size = sum(p.size for p in parts)

    def read(self, pos, length):
        out = b""
        while length > 0:
            i = max(j for j, s in enumerate(self.starts) if s <= pos)
            n = min(length, self.parts[i].size - (pos - self.starts[i]))
            out += self.parts[i].read(pos - self.starts[i], n)
            pos, length = pos + n, length - n
        return out


class Window:
    """Bytes [start, start + size) of another source."""

    def __init__(self, src, start, size):
        self.src, self.start, self.size = src, start, size

    def read(self, pos, length):
        return self.src.read(self.start + pos, length)


def _u(fmt, b, off):
    return struct.unpack_from(fmt, b, off)[0]


class RemoteZip:
    def __init__(self, src, disk_starts=(0,)):
        self.src, self.disk_starts = src, list(disk_starts)
        self.members = self._read_central_directory()

    @classmethod
    def from_urls(cls, urls):
        c = Concat([HttpFile(u) for u in urls])
        return cls(c, c.starts)

    @classmethod
    def nested(cls, outer, name):
        method, csize, usize, _, _ = outer.members[name]
        if method != 0:
            raise ValueError(f"{name} is compressed; only stored nested zips can be range-read")
        return cls(Window(outer.src, outer.data_offset(name), usize))

    def _read_central_directory(self):
        tail_len = min(self.src.size, 65536 + 22)
        tail = self.src.read(self.src.size - tail_len, tail_len)
        e = tail.rfind(b"PK\x05\x06")
        if e < 0:
            raise ValueError("no end-of-central-directory record")
        _, _, cd_disk, _, n_entries, cd_size, cd_off, _ = struct.unpack_from("<IHHHHIIH", tail, e)
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        if loc >= 0:  # zip64
            _, z64_disk, z64_off, _ = struct.unpack_from("<IIQI", tail, loc)
            rec = self.src.read(self.disk_starts[z64_disk] + z64_off, 56)
            cd_disk, n_entries, cd_size, cd_off = _u("<I", rec, 20), _u("<Q", rec, 32), _u("<Q", rec, 40), _u("<Q", rec, 48)
        cd = self.src.read(self.disk_starts[cd_disk] + cd_off, cd_size)
        members, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, csize, usize = _u("<H", cd, p + 10), _u("<I", cd, p + 20), _u("<I", cd, p + 24)
            nlen, xlen, clen, disk = _u("<H", cd, p + 28), _u("<H", cd, p + 30), _u("<H", cd, p + 32), _u("<H", cd, p + 34)
            loff = _u("<I", cd, p + 42)
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            q, end = p + 46 + nlen, p + 46 + nlen + xlen
            while q + 4 <= end:  # zip64 extended information
                tag, ln = struct.unpack_from("<HH", cd, q)
                if tag == 1:
                    r = q + 4
                    if usize == 0xFFFFFFFF:
                        usize, r = _u("<Q", cd, r), r + 8
                    if csize == 0xFFFFFFFF:
                        csize, r = _u("<Q", cd, r), r + 8
                    if loff == 0xFFFFFFFF:
                        loff, r = _u("<Q", cd, r), r + 8
                    if disk == 0xFFFF:
                        disk = _u("<I", cd, r)
                    break
                q += 4 + ln
            members[name] = (method, csize, usize, disk, loff)
            p = end + clen
        if len(members) != n_entries:
            raise ValueError(f"central directory: parsed {len(members)} of {n_entries} entries")
        return members

    def data_offset(self, name):
        """Position of the member's (compressed) data in self.src."""
        _, _, _, disk, loff = self.members[name]
        pos = self.disk_starts[disk] + loff
        head = self.src.read(pos, 30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"bad local header for {name}")
        return pos + 30 + _u("<H", head, 26) + _u("<H", head, 28)

    def read(self, name):
        method, csize, _, disk, loff = self.members[name]
        pos = self.disk_starts[disk] + loff
        guess = 30 + len(name.encode()) + 64  # local header + name + typical extra field, in the same request
        blob = self.src.read(pos, min(guess + csize, self.src.size - pos))
        if blob[:4] != b"PK\x03\x04":
            raise ValueError(f"bad local header for {name}")
        start = 30 + _u("<H", blob, 26) + _u("<H", blob, 28)
        data = blob[start:start + csize]
        if len(data) < csize:
            data += self.src.read(pos + start + len(data), csize - len(data))
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise ValueError(f"unsupported compression {method} for {name}")
