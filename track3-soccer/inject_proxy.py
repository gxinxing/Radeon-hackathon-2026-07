#!/usr/bin/env python3
"""Complete setup: install deps + start VNC + inject proxy into running Jupyter via gdb.
Run: /opt/venv/bin/python3 /workspace/inject_proxy.py
"""
import subprocess, sys, os, time

# 1. Install proxy if missing
r = subprocess.run(["/opt/venv/bin/python3", "-c", "import jupyter_server_proxy"], capture_output=True)
if r.returncode != 0:
    print("Installing jupyter-server-proxy...")
    subprocess.run(["/opt/venv/bin/pip", "install", "-q", "jupyter-server-proxy"], capture_output=True)

# 2. Install apt packages if missing
print("Installing system packages...")
subprocess.run("apt-get update -qq 2>/dev/null; apt-get install -y -qq gdb tigervnc-standalone-server openbox novnc websockify iproute2 xdg-utils libasound2t64 libgl1 libglx-mesa0 libegl1 libgbm1 libdrm2 libxkbcommon0 libwayland-client0 libvulkan1 libnotify4 libatspi2.0-0 libsecret-1-0 libgtk-3-0 libnss3 libnspr4 libxss1 mesa-utils dbus-x11 2>&1 | tail -3", shell=True, capture_output=True)

# 3. Install Booster Studio if missing
if not os.path.exists("/usr/share/booster-studio/booster-studio"):
    print("Installing Booster Studio...")
    subprocess.run("wget -q -O /tmp/bs.deb 'https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb' && dpkg -i /tmp/bs.deb 2>&1 | tail -3; apt-get install -f -y -qq 2>&1 | tail -3", shell=True, capture_output=True)

# 4. Start VNC + Booster Studio + websockify if not running
if not subprocess.run("pgrep Xtigervnc", shell=True, capture_output=True).stdout:
    print("Starting VNC...")
    os.makedirs("/root/.vnc", exist_ok=True)
    subprocess.run("echo ***REMOVED*** | vncpasswd -f > /root/.vnc/passwd && chmod 600 /root/.vnc/passwd", shell=True)
    subprocess.run("rm -f /tmp/.X99-lock /tmp/.X11-unix/X99", shell=True)
    subprocess.run("nohup Xtigervnc :99 -localhost no -rfbport 5999 -PasswordFile /root/.vnc/passwd -geometry 1920x1080 -depth 24 -SecurityTypes VncAuth >/tmp/vnc.log 2>&1 &", shell=True)
    time.sleep(3)
    env = os.environ.copy()
    env["DISPLAY"] = ":99"
    subprocess.Popen(["openbox"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    subprocess.Popen(["/usr/share/booster-studio/booster-studio", "--no-sandbox", "--disable-gpu-sandbox"], env=env, stdout=open("/tmp/bs.log","w"), stderr=subprocess.STDOUT)
    time.sleep(3)
    subprocess.run("nohup websockify --web=/usr/share/novnc/ 6080 localhost:5999 &>/tmp/ws.log 2>&1 &", shell=True)
    time.sleep(2)

# Check ports
r = subprocess.run("ss -tlnp | grep -E '5999|6080'", shell=True, capture_output=True, text=True)
print(f"PORTS:\n{r.stdout}")

# 5. Inject proxy into running Jupyter via gdb
jupyter_pid = subprocess.run("pgrep -f jupyter-lab | head -1", shell=True, capture_output=True, text=True).stdout.strip()
if not jupyter_pid:
    print("ERROR: No Jupyter process found")
    sys.exit(1)

print(f"Jupyter PID: {jupyter_pid}")

# Write the injection Python code that will run inside the Jupyter process
inject_code = '''import sys
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")
try:
    from jupyter_server.proxy import ProxyHandler
    from jupyter_server.serverapp import ServerApp
    app = ServerApp.instance()
    web_app = app.web_app
    base_url = web_app.settings.get("base_url", "/")
    pattern = base_url + r"proxy/(\\d+)(/.*)"
    web_app.add_handlers(".*$", [(pattern, ProxyHandler)])
    with open("/tmp/inject_result.txt", "w") as f:
        f.write("PROXY_INJECTED_OK")
except Exception as e:
    with open("/tmp/inject_result.txt", "w") as f:
        f.write(f"PROXY_INJECT_ERROR: {e}\\n")
        import traceback
        f.write(traceback.format_exc())
'''

with open("/tmp/inject_code.py", "w") as f:
    f.write(inject_code)

# Use gdb to inject the code into the running Jupyter process
gdb_script = f"""set pagination off
attach {jupyter_pid}
call (int)PyGILState_Ensure()
call (int)PyRun_SimpleString("exec(open('/tmp/inject_code.py').read())")
call (void)PyGILState_Release(0)
detach
quit
"""

with open("/tmp/gdb_script.txt", "w") as f:
    f.write(gdb_script)

print("Running gdb injection...")
r = subprocess.run(["gdb", "-batch", "-x", "/tmp/gdb_script.txt"], capture_output=True, text=True, timeout=30)
# Show last part of gdb output
stdout_lines = r.stdout.strip().split('\n')
for line in stdout_lines[-10:]:
    print(f"GDB: {line}")
stderr_lines = r.stderr.strip().split('\n')
for line in stderr_lines[-5:]:
    print(f"GDB_ERR: {line}")

# Check result
time.sleep(2)
try:
    with open("/tmp/inject_result.txt") as f:
        result = f.read()
    print(f"INJECTION RESULT: {result}")
except Exception as e:
    print(f"ERROR reading result: {e}")
