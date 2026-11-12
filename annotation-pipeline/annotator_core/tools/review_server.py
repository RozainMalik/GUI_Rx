"""A local reviewer for the boxes, because the model cannot judge its own.

One keypress judges a row -
C correct, L loose, W wrong - and that is the fast path everything else is arranged around: the
next screenshot is preloaded, the verdict is written before the view advances, and nothing needs
the mouse unless a box is actually wrong (drag to redraw, Enter saves it as "corrected").

Reads a grading sample (work/review/sample_AB.json, built by `annotator grade`) and the local
copy of the screenshots. Verdicts append to work/review/verdicts.jsonl keyed by image + box, so
re-running skips what is already judged and a box that changed after a replay is queued again.

    python tools/review_server.py --sample work/review/sample_AB.json --root path/to/screenshots
"""
import argparse
import json
import mimetypes
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotator.grading import VERDICTS, append_verdict, load_verdicts, verdict_key  # noqa: E402


def load_items(sample_file, root, only_local=True):
    """The sampled rows; by default only those whose screenshot is on this machine, because a
    queue of pictures it cannot show would spend a reviewer's attention on 404s."""
    items = json.loads(Path(sample_file).read_text(encoding="utf-8"))["items"]
    if not only_local:
        return items
    present = [it for it in items if (Path(root) / it["image"]).is_file()]
    if len(present) < len(items):
        print(f"{len(items) - len(present)} sampled screenshots are not on this machine - skipped")
    return present


class Handler(BaseHTTPRequestHandler):
    items = []
    verdicts = {}
    root = Path(".")
    verdict_path = Path("work/review/verdicts.jsonl")

    def log_message(self, *args):
        pass                                   # the console is for progress

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if route == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if route == "/api/items":
            payload = []
            for it in self.items:
                v = self.verdicts.get(verdict_key(it["image"], it["bbox_px"]))
                payload.append(dict(it, reviewed=v is not None, verdict=v["verdict"] if v else None))
            return self._send(200, json.dumps({"items": payload, "reviewed": sum(p["reviewed"] for p in payload)}))
        if route.startswith("/shot/"):
            target = (self.root / route[len("/shot/"):]).resolve()
            if not str(target).startswith(str(self.root.resolve())):      # never walk out of the dataset
                return self._send(403, json.dumps({"error": "outside root"}))
            if not target.is_file():
                return self._send(404, json.dumps({"error": "no such screenshot"}))
            return self._send(200, target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        return self._send(404, json.dumps({"error": "no such route"}))

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/verdict":
            return self._send(404, json.dumps({"error": "no such route"}))
        length = int(self.headers.get("Content-Length") or 0)
        try:
            verdict = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, json.dumps({"error": "bad json"}))
        if (not verdict.get("image") or verdict.get("verdict") not in VERDICTS
                or not verdict.get("original_bbox_px") or not verdict.get("bbox_px")):
            return self._send(400, json.dumps({"error": "bad verdict"}))
        append_verdict(self.verdict_path, verdict)             # before the view advances
        self.verdicts[verdict_key(verdict["image"], verdict["original_bbox_px"])] = verdict
        print(f"  {len(self.verdicts):>5}/{len(self.items)}  {verdict['verdict']:<10} {verdict['image']}")
        return self._send(200, json.dumps({"ok": True, "reviewed": len(self.verdicts)}))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>MedSPOT box review</title>
<style>
  :root { --bg:#141417; --panel:#1c1c21; --fg:#eee; --dim:#9a9aa4; --box:#ff2020;
          --ok:#39d353; --warn:#ffb020; --bad:#ff5c5c; --fix:#4aa3ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }
  header { display:flex; gap:18px; align-items:center; padding:8px 14px; background:var(--panel);
           border-bottom:1px solid #2a2a31; position:sticky; top:0; }
  #bar { flex:1; height:8px; background:#2a2a31; border-radius:4px; overflow:hidden; }
  #fill { height:100%; width:0; background:var(--ok); transition:width .15s; }
  #stage { position:relative; margin:0 auto; line-height:0; }
  #shot { max-width:100%; display:block; }
  #overlay { position:absolute; inset:0; cursor:crosshair; }
  #meta { padding:10px 14px 40px; max-width:1400px; margin:0 auto; }
  .kind { color:var(--dim); }
  .v { margin:6px 0 0 14px; }
  .vlabel { color:var(--box); }
  kbd { background:#2a2a31; border:1px solid #3a3a44; border-radius:4px; padding:1px 6px; font:12px ui-monospace,monospace; }
  #help { color:var(--dim); padding:0 14px 20px; max-width:1400px; margin:0 auto; }
  .tag { padding:2px 8px; border-radius:10px; font-size:12px; }
  .correct { background:#12351d; color:var(--ok); }
  .loose { background:#3a2c10; color:var(--warn); }
  .wrong { background:#3a1717; color:var(--bad); }
  .corrected { background:#102a44; color:var(--fix); }
  #done { text-align:center; padding:80px 20px; font-size:18px; }
</style></head><body>
<header>
  <b id="pos">-</b>
  <div id="bar"><div id="fill"></div></div>
  <span id="count" class="kind">-</span>
  <span id="state"></span>
</header>
<div id="stage"><img id="shot" alt=""><canvas id="overlay"></canvas></div>
<div id="meta"></div>
<div id="help">
  <kbd>C</kbd>/<kbd>Space</kbd> correct &middot; <kbd>L</kbd> loose &middot; <kbd>W</kbd> wrong &middot;
  drag on the image to redraw, <kbd>Enter</kbd> saves it as corrected &middot;
  <kbd>Z</kbd> undo drawn box &middot; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move &middot; <kbd>U</kbd> next unreviewed
</div>
<script>
let items = [], i = 0, drawn = null, drag = null, reviewedCount = 0;
const img = document.getElementById('shot'), cv = document.getElementById('overlay');
const ctx = cv.getContext('2d');

async function boot() {
  const r = await (await fetch('/api/items')).json();
  items = r.items; reviewedCount = r.reviewed;
  if (!items.length) { document.body.innerHTML = '<div id="done">Nothing to review: no sampled screenshots found locally.</div>'; return; }
  i = Math.max(0, items.findIndex(x => !x.reviewed));
  show();
}

function show() {
  const it = items[i];
  drawn = null;
  img.onload = () => { sizeCanvas(); paint(); preload(); };
  img.src = '/shot/' + it.image.split('/').map(encodeURIComponent).join('/');
  document.getElementById('pos').textContent = (i + 1) + ' / ' + items.length;
  document.getElementById('count').textContent = reviewedCount + ' reviewed';
  document.getElementById('fill').style.width = (100 * reviewedCount / items.length) + '%';
  const st = document.getElementById('state');
  st.className = 'tag ' + (it.verdict || '');
  st.textContent = it.verdict || '';
  document.getElementById('meta').innerHTML =
      '<b>' + esc(it.app) + '</b> / ' + esc(it.frame) + ' <span class="kind">tier ' + esc(it.tier) + '</span>' +
      '<div class="kind">' + esc(it.element_kind) + ': ' + esc(it.element) + '</div>' +
      ['functional','visual','spatial'].map(v =>
        '<div class="v"><span class="vlabel">' + v + ':</span> ' + esc(it.instructions[v] || '') + '</div>').join('');
}

function esc(s) { return (s || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }
function preload() { const n = items[i + 1]; if (n) { const p = new Image();
  p.src = '/shot/' + n.image.split('/').map(encodeURIComponent).join('/'); } }

function sizeCanvas() {
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  cv.style.width = img.clientWidth + 'px'; cv.style.height = img.clientHeight + 'px';
}
function scale() { return img.clientWidth / img.naturalWidth; }

function paint() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  const s = scale(), b = drawn || items[i].bbox_px;
  ctx.lineWidth = 2;
  ctx.strokeStyle = drawn ? '#ffb020' : '#ff2020';
  ctx.strokeRect(b[0]*s, b[1]*s, (b[2]-b[0])*s, (b[3]-b[1])*s);
}

cv.addEventListener('mousedown', e => {
  const r = cv.getBoundingClientRect();
  drag = [(e.clientX-r.left)/scale(), (e.clientY-r.top)/scale()];
});
cv.addEventListener('mousemove', e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  const x = (e.clientX-r.left)/scale(), y = (e.clientY-r.top)/scale();
  drawn = [Math.round(Math.min(drag[0],x)), Math.round(Math.min(drag[1],y)),
           Math.round(Math.max(drag[0],x)), Math.round(Math.max(drag[1],y))];
  paint();
});
window.addEventListener('mouseup', () => { drag = null; });
window.addEventListener('resize', () => { sizeCanvas(); paint(); });

async function send(verdict, bbox) {
  const it = items[i];
  const body = {image: it.image, verdict: verdict, bbox_px: bbox || it.bbox_px, original_bbox_px: it.bbox_px, tier: it.tier};
  const r = await (await fetch('/api/verdict', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})).json();
  if (!it.reviewed) reviewedCount = r.reviewed;
  it.reviewed = true; it.verdict = verdict;
  next();
}
function next() { if (i < items.length - 1) { i++; } show(); }
function prev() { if (i > 0) { i--; show(); } }
function nextUnreviewed() { const j = items.findIndex((x, k) => k > i && !x.reviewed); if (j >= 0) { i = j; show(); } }

document.addEventListener('keydown', e => {
  if (e.key === 'c' || e.key === 'C' || e.key === ' ') { e.preventDefault(); send('correct'); }
  else if (e.key === 'l' || e.key === 'L') { send('loose'); }
  else if (e.key === 'w' || e.key === 'W') { send('wrong'); }
  else if (e.key === 'Enter') { if (drawn) send('corrected', drawn); }
  else if (e.key === 'z' || e.key === 'Z') { drawn = null; paint(); }
  else if (e.key === 'ArrowRight') { next(); }
  else if (e.key === 'ArrowLeft') { prev(); }
  else if (e.key === 'u' || e.key === 'U') { nextUnreviewed(); }
});
boot();
</script></body></html>
"""


def make_server(items, root, verdict_path, port=8777):
    Handler.items = items
    Handler.verdicts = load_verdicts(verdict_path)
    Handler.root = Path(root)
    Handler.verdict_path = Path(verdict_path)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(sample_file, root, verdict_path, port=8777, open_browser=True, only_local=True) -> int:
    items = load_items(sample_file, root, only_local)
    if not items:
        print("no sampled screenshots found locally - check --sample and --root")
        return 1
    server = make_server(items, root, verdict_path, port)
    apps = sorted({it["app"] for it in items})
    print(f"{len(items)} rows across {len(apps)} app(s); {len(Handler.verdicts)} already reviewed")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"verdicts -> {verdict_path}\nopen {url}   (ctrl-c to stop; progress is saved as you go)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(Handler.verdicts)}/{len(items)} reviewed.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="work/review/sample_AB.json")
    parser.add_argument("--root", default="E:/MedSpot/Annotation Pipeline/MedSPOT Training Data",
                        help="local folder holding <App>/<frame>.png")
    parser.add_argument("--out", default="work/review/verdicts.jsonl")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--all-apps", action="store_true", help="queue rows whose screenshot is not on this machine")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    return serve(args.sample, args.root, args.out, args.port, not args.no_open, not args.all_apps)


if __name__ == "__main__":
    raise SystemExit(main())
