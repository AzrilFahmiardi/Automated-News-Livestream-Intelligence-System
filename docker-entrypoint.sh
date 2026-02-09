#!/bin/bash
set -e

echo "=========================================="
echo "News Livestream Intelligence System"
echo "Starting services..."
echo "=========================================="

# ============================================================
# 1. Start Xvfb (Virtual Framebuffer)
# ============================================================
echo "[1/3] Starting Xvfb virtual display..."

# Kill any existing Xvfb process
pkill Xvfb 2>/dev/null || true
sleep 1

# Clean up stale X11 lock files and sockets
rm -f /tmp/.X99-lock 2>/dev/null || true
rm -rf /tmp/.X11-unix/X99 2>/dev/null || true

# Start Xvfb on display :99 with 1920x1080 resolution, 24-bit color
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to start
sleep 2

# Verify Xvfb is running
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi

export DISPLAY=:99
echo "    Xvfb started on display :99 (PID: $XVFB_PID)"

# ============================================================
# 2. Start PulseAudio with Virtual Sink
# ============================================================
echo "[2/3] Starting PulseAudio with virtual audio sink..."

# Create PulseAudio runtime directory
mkdir -p /tmp/pulse /run/pulse ~/.config/pulse
chmod 755 /tmp/pulse /run/pulse

# Kill any existing PulseAudio process
pulseaudio --kill 2>/dev/null || true
pkill -9 pulseaudio 2>/dev/null || true
sleep 1

# Create a minimal PulseAudio config for container use
cat > ~/.config/pulse/default.pa << 'EOF'
# Minimal PulseAudio config for Docker container
load-module module-native-protocol-unix auth-anonymous=1
load-module module-null-sink sink_name=virtual_speaker sink_properties=device.description="Virtual_Speaker"
set-default-sink virtual_speaker
EOF

# Create client config to use local socket
cat > ~/.config/pulse/client.conf << 'EOF'
default-server = unix:/tmp/pulse/native
autospawn = no
EOF

# Set environment 
export PULSE_RUNTIME_PATH=/tmp/pulse
export PULSE_STATE_PATH=/tmp/pulse

# Start PulseAudio in user mode 
pulseaudio \
    --daemonize=true \
    --disallow-exit \
    --exit-idle-time=-1 \
    --use-pid-file=false \
    --log-target=stderr \
    --log-level=error \
    -F ~/.config/pulse/default.pa \
    2>/dev/null

# Wait for PulseAudio to initialize
sleep 2

# Check if PulseAudio is running
if pgrep -x pulseaudio > /dev/null; then
    DEFAULT_SINK=$(pactl get-default-sink 2>/dev/null || echo "unknown")
    if [ "$DEFAULT_SINK" = "virtual_speaker" ]; then
        echo "    PulseAudio started with default sink: $DEFAULT_SINK"
    else
        echo "    WARNING: Virtual speaker not set as default, attempting to configure..."
        pactl load-module module-null-sink sink_name=virtual_speaker sink_properties=device.description="Virtual_Speaker" 2>/dev/null || true
        pactl set-default-sink virtual_speaker 2>/dev/null || true
        DEFAULT_SINK=$(pactl get-default-sink 2>/dev/null || echo "unknown")
        echo "    PulseAudio default sink: $DEFAULT_SINK"
    fi
else
    echo "    WARNING: PulseAudio failed to start, audio recording will not work"
    DEFAULT_SINK="unavailable"
fi

# ============================================================
# 3. Display System Info
# ============================================================
echo "[3/3] System ready!"
echo ""
echo "Environment:"
echo "    DISPLAY=$DISPLAY"
echo "    Audio Sink=$DEFAULT_SINK"
echo "    Python=$(python --version 2>&1)"
echo ""
echo "=========================================="
echo "Starting application..."
echo "=========================================="
echo ""

# ============================================================
# 4. Cleanup handler
# ============================================================
cleanup() {
    echo ""
    echo "Shutting down..."
    
    # Stop PulseAudio
    pulseaudio --kill 2>/dev/null || true
    
    # Stop Xvfb
    kill $XVFB_PID 2>/dev/null || true
    
    echo "Cleanup complete"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT SIGQUIT

# ============================================================
# 5. Execute the main command
# ============================================================
exec "$@"
