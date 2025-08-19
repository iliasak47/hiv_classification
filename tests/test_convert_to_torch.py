import sys, subprocess
from pathlib import Path
import torch
import torch.nn as nn


def locate_script() -> Path:
    candidates = [
        Path("src/models_/convert_to_torch.py"),  
        Path("src/models/convert_to_torch.py"),
        Path("convert_to_torch.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("convert_to_torch.py not found in expected locations.")


def build_state_dict(models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.fc3 = nn.Linear(7,32), nn.Linear(32,16), nn.Linear(16,1)
        def forward(self, x):
            x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)
    m = Net().eval()
    torch.save(m.state_dict(), models_dir / "pytorch_mlp.pt")


def test_convert_script_creates_torchscript(tmp_path: Path):
    # Locate script and detect which models dir it expects by reading its content
    script_src = locate_script()
    code = script_src.read_text(encoding="utf-8")
    models_dirname = "models_" if "models_" in code else "models"

    # Arrange tmp workspace with expected directory and a state_dict
    models_dir = tmp_path / models_dirname
    build_state_dict(models_dir)

    # Copy script into tmp to run relative paths
    script_dst = tmp_path / "convert_to_torch.py"
    script_dst.write_text(code, encoding="utf-8")

    # Act: run the script in tmp workspace
    res = subprocess.run([sys.executable, str(script_dst)], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    # Assert: TorchScript file exists and is loadable
    ts_path = models_dir / "pytorch_mlp_ts.pt"
    assert ts_path.exists()
    ts = torch.jit.load(str(ts_path), map_location="cpu")
    ts.eval()
    out = ts(torch.zeros(4, 7))
    assert out.shape[0] == 4
