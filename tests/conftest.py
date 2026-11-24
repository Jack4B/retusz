import os, sys, tempfile
from pathlib import Path
os.environ["RETUSZ_HOME"] = tempfile.mkdtemp(prefix="retusz-test-")  # never touch the real user data
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pytest

@pytest.fixture
def rng():
    return np.random.default_rng(0)

def synth(h=240, w=360, seed=0, tint=(1.0, 1.0, 1.0)):
    """Smooth synthetic photo: gradients + blobs, optional colour tint."""
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([x/w, y/h, 0.5+0.3*np.sin(x/30+y/50)])
    for _ in range(6):
        cy, cx, rad = r.uniform(0, h), r.uniform(0, w), r.uniform(15, 60)
        img += (np.exp(-((x-cx)**2+(y-cy)**2)/(2*rad**2))[..., None]*r.uniform(-0.4, 0.4, 3))
    img = np.clip(img*np.array(tint), 0, 1)
    return (img*255).astype(np.uint8)

REAL_SRC = Path("/workspace/retouch-proto/input/photo_a.jpg")
REAL_REF = Path("/workspace/retouch-proto/input/photo_b.jpg")
