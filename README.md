# ComfyUi-Untwisting-RoPE-Klein9b

100% vibe-coded, appears to work. If you get too much structure bleeding in from reference increase `start_block`. Other very effectual parameters seem to be `gamma` on the RF node and `adain_strength` on the RoPE node.

<img width="2458" height="1385" alt="Klein 9b Example" src="https://github.com/user-attachments/assets/9da4174c-233f-4349-bd9b-f9a38d160c5d" />


# ComfyUi-Untwisting-RoPE (Training-Free Style Tranfer)

This is a ComfyUi implementation of [Untwisting RoPE: Frequency Control for Shared Attention in DiTs](https://arxiv.org/abs/2602.05013)

https://untwisting-rope.github.io/

<img width="1280" alt="Image 4" src="https://github.com/user-attachments/assets/ea4b7f49-45b0-4812-845b-0bd58824f27c" />


- Only Z-image Turbo is implemented so far (I'll see if I'll implement other models).
- You can see more examples [here](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE/tree/main/Examples).

## Installation

Navigate to the **ComfyUI\custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUi-Untwisting-RoPE
```

Navigate to the **ComfyUI\custom_nodes\ComfyUi-ImageGalleryLoader** folder, open cmd and run:

```bash
..\..\..\python_embeded\python.exe -s -m pip install -r "requirements.txt"
```
Restart ComfyUI after installation.

## Usage

Here's a [workflow](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE/blob/main/Workflow_zimage_turbo.json) for those interested.
You also need this [custom node](https://github.com/BigStationW/ComfyUi-Scale-Image-to-Total-Pixels-Advanced) to make it work.
